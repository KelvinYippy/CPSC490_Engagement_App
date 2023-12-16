# CPSC490 - EngagementApp Server [Running the Server]

This folder contains all the necessary code and functions for running the EngagementApp Server.

## Step Zero [Python Tools]

This is a python heavy repository, so the following tools may be of use:

[Python3](https://www.python.org/downloads/)
[Pip3](https://www.activestate.com/resources/quick-reads/how-to-install-and-use-pip3/)

```bash
python3 -m pip3 install --upgrade pip
```

## Step One [Virtual Environment]

Make sure you create a virtual environment for this folder. 

```bash
python3 -m venv .
```

If the above command does not work, you may use the [following link](https://docs.python.org/3/library/venv.html) to assist you.

Once the virtual environment is created, activate the virtual environment by running the following command, assuming you are still in the model folder.

```bash
source ./bin/activate
```

## Step Two [Packages]

Now, you will need to download the required packages with the following command:

```bash
pip3 install -r ./requirements.txt
```

If this does not work, the following packages should at least be required to run the files in this folder:
```bash
pip3 install uvicorn
pip3 install fastapi
pip3 install torch
pip3 install opencv-python
pip3 install python-multipart
```

If that still does not work, here is an exhaustive list of packages needed:
- torch
- uvicorn
- opencv-python
- fastapi
- python-multipart

## Step Three [Run the Server]

To run the API, run the following command. The web-client ONLY WORKS if BOTH THE CLIENT AND SERVER ARE ACTIVE AT THE SAME TIME!

```bash
uvicorn main:app --reload
```

## Step Four [Updating the Model]

```python
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
```

```python
# Load the PyTorch model
# Replace 'YourModel' and 'model.pth' with your model class and file
model = VideoClassifier(2)
model.load_state_dict(torch.load('./model_0024_weights.pt')['state_dict'])
model.eval()
```

The first code block loads the class architecture that the class uses, while the second code block loads weights. If you ever want to change the architecture / weights, here are the places to do so.