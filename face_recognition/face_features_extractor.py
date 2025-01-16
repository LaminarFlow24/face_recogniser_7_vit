import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModel
from facenet_pytorch import MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face


class FaceFeaturesExtractor:
    def __init__(self):
        # Load ViTFace feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("WangYueFt/vit-face-masked")
        self.vitface = AutoModel.from_pretrained("WangYueFt/vit-face-masked").eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vitface = self.vitface.to(self.device)

        # Face aligner (e.g., MTCNN for face detection and alignment)
        self.aligner = MTCNN(keep_all=True, thresholds=[0.6, 0.7, 0.9])

    def extract_features(self, img):
        # Detect and align faces
        bbs, _ = self.aligner.detect(img)
        if bbs is None:
            return None, None

        # Extract faces and prepare for ViTFace
        faces = [self._extract_face(img, bb) for bb in bbs]
        faces = [self.feature_extractor(images=face, return_tensors="pt") for face in faces]

        # Stack and send faces to device
        face_tensors = torch.cat([face["pixel_values"].to(self.device) for face in faces], dim=0)

        # Extract embeddings using ViTFace
        with torch.no_grad():
            embeddings = self.vitface(face_tensors).last_hidden_state[:, 0, :].cpu().numpy()

        return bbs, embeddings

    def _extract_face(self, img, bb):
        """
        Extracts and crops face from the image based on bounding box.
        """
        return extract_face(img, bb)

    def __call__(self, img):
        return self.extract_features(img)
