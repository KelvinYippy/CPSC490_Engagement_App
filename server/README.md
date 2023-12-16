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