import os
import shutil
from typing import Any
import insightface
import threading

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        with FACE_ANALYSER_LOCK:
            if FACE_ANALYSER is None:
                # CoreML EP has shape inference bugs with det_10g.onnx — force CPU for detection
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition']
                )
                # Use user's execution providers for recognition model only
                if 'recognition' in FACE_ANALYSER.models and modules.globals.execution_providers != ['CPUExecutionProvider']:
                    FACE_ANALYSER.models['recognition'].session.set_providers(
                        modules.globals.execution_providers
                    )

                FACE_ANALYSER.prepare(ctx_id=0, det_size=(320, 320))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None


class FaceTracker:
    def __init__(self):
        self.prev_gray = None
        self.tracked_faces = []

    def update(self, frame: np.ndarray, detected_faces: list = None) -> list:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If new detections are provided, or if we have no tracked faces, reset tracking
        if detected_faces is not None or not self.tracked_faces or self.prev_gray is None:
            self.prev_gray = curr_gray
            self.tracked_faces = []
            if detected_faces:
                from insightface.app.common import Face
                for face in detected_faces:
                    if face is not None and hasattr(face, 'kps') and face.kps is not None:
                        cloned_face = Face()
                        for k, v in face.items():
                            if hasattr(v, 'copy'):
                                cloned_face[k] = v.copy()
                            else:
                                cloned_face[k] = v
                        self.tracked_faces.append(cloned_face)
            return self.tracked_faces

        # Perform tracking of existing faces
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        next_tracked_faces = []
        for face in self.tracked_faces:
            if not hasattr(face, 'kps') or face.kps is None:
                continue
            p0 = face.kps.reshape(-1, 1, 2).astype(np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, p0, None, **lk_params)
            st = st.reshape(-1)
            if np.sum(st) >= 3:
                # Successfully tracked
                tracked_p0 = p0.reshape(-1, 2)[st == 1]
                tracked_p1 = p1.reshape(-1, 2)[st == 1]
                displacement = tracked_p1 - tracked_p0
                mean_displacement = np.mean(displacement, axis=0)
                
                # Update kps
                new_kps = face.kps.copy()
                for i in range(5):
                    if st[i] == 1:
                        new_kps[i] = p1[i, 0]
                    else:
                        new_kps[i] = face.kps[i] + mean_displacement
                
                # Update bbox
                new_bbox = face.bbox.copy()
                new_bbox[0] += mean_displacement[0]
                new_bbox[1] += mean_displacement[1]
                new_bbox[2] += mean_displacement[0]
                new_bbox[3] += mean_displacement[1]
                
                face.kps = new_kps
                face.bbox = new_bbox
                next_tracked_faces.append(face)
            else:
                # Tracking lost for this face, drop it
                pass
                
        self.prev_gray = curr_gray
        self.tracked_faces = next_tracked_faces
        return self.tracked_faces


def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)
            
            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            modules.globals.source_target_map[i]['target_faces_in_frame'] = temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1