from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import torch
import torch.nn as nn
from os import remove
from cv2 import VideoCapture, CAP_PROP_FPS, destroyAllWindows, resize

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(VideoClassifier, self).__init__()
        self.conv1 = nn.Conv2d(15, 32, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm_2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * (160 // 8) * (90 // 8), 256)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm_1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.batch_norm_2(self.conv2(x))))
        x = self.pool(nn.functional.relu(self.batch_norm_3(self.conv3(x))))
        x = x.reshape(-1, 64 * (160 // 8) * (90 // 8))  # Adjust input size
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Load the PyTorch model
# Replace 'YourModel' and 'model.pth' with your model class and file
model = VideoClassifier(2)
model.load_state_dict(torch.load('./model_0024_weights.pt')['state_dict'])
model.eval()

def get_frame_rate(path: str):
    """
        Get the frame rate per second of a video given its path
        path : str
            The path of where the video is located
    """
    cap = VideoCapture(path)
    fps = int(cap.get(CAP_PROP_FPS))
    return fps

def extract_frames(path: str) -> torch.Tensor:
    """
        Extract frames from the video located at path, and puts it into frames folder of boolean type.
        path : str
            The path of where the video is located
        boolean: str
            The classification of the video (Available Options: ["Yes", "No"])
    """
    cap = VideoCapture(path)
    i, frames = 0, 0
    frame_rate = get_frame_rate(path)
    frame_lst = []
    while cap.isOpened() and frames < 5:
        ret, frame = cap.read()
        if ret == False:
            break
        if i % (frame_rate // 4) == 0:
            frame = resize(frame, (160, 90))
            frame = torch.from_numpy(frame).to(torch.uint8)
            frame = frame.permute(2, 0, 1)
            frame_lst.append(frame)
            frames += 1
        i += 1
    cap.release()
    destroyAllWindows()
    return torch.cat(frame_lst, dim=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp = NamedTemporaryFile(delete=False)
    try:
        contents = await file.read()
        with temp as f:
            f.write(contents)
        file.close()
        frames = extract_frames(temp.name)
        frames = frames.unsqueeze(0)
        with torch.no_grad():
            outputs = model(frames.float())
        print(outputs)
        remove(temp.name)
        _, predicted = torch.max(outputs, 1)
        return JSONResponse(content={'prediction': predicted.tolist()})
    except Exception as e:
        print(e)
        return JSONResponse(content={'error': str(e)})