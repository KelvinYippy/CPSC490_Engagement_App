import React, { ChangeEvent, useEffect, useRef, useState } from 'react';
import { PredictService } from './PredictService';
import './App.css'

const mimeType = 'video/webm; codecs="opus,vp8"';

const App = () => {
  
	const [permission, setPermission] = useState(false);

	const mediaRecorder = useRef(null);

	const liveVideoFeed = useRef(null);

	const [recordingStatus, setRecordingStatus] = useState("inactive");

	const [stream, setStream] = useState<MediaStream | null>(null);

	const [recordedVideo, setRecordedVideo] = useState("");

	const [videoChunks, setVideoChunks] = useState<any[]>([]);

  const [prediction, setPrediction] = useState<number | null>(null)

  const [isVideoRecord, setIsVideoRecord] = useState(true)

  useEffect(() => {
    const test = async () => {
      const predictService = new PredictService()
      const response = await predictService.predictVideo(recordedVideo)
      const r_json = await response.json()
      setPrediction(r_json.prediction[0])
    }
    if (recordedVideo.length > 0) 
      test()
  }, [recordedVideo])

	const getCameraPermission = async () => {
		setRecordedVideo("");
		//get video and audio permissions and then stream the result media stream to the videoSrc variable
		if ("MediaRecorder" in window) {
			try {
				const videoConstraints = {
					audio: false,
					video: true,
				};
				const audioConstraints = { audio: true };

				// create audio and video streams separately
        const [audioStream, videoStream] = await Promise.all([
          navigator.mediaDevices.getUserMedia(audioConstraints),
          navigator.mediaDevices.getUserMedia(videoConstraints)
        ])

				setPermission(true);
        setPrediction(null);

				//combine both audio and video streams

				const combinedStream = new MediaStream([
					...videoStream.getVideoTracks(),
					...audioStream.getAudioTracks(),
				]);

				setStream(combinedStream);

				//set videostream to live feed player
				(liveVideoFeed.current as any).srcObject = videoStream;
			} catch (err) {
				alert(err);
			}
		} else {
			alert("The MediaRecorder API is not supported in your browser.");
		}
	};

	const startRecording = async () => {
		setRecordingStatus("recording");

		const media = new MediaRecorder(stream!, { mimeType });

		(mediaRecorder.current as any) = media;

		(mediaRecorder.current as any).start();

		let localVideoChunks: any[] = [];

		(mediaRecorder.current as any).ondataavailable = (event: any) => {
			if (typeof event.data === "undefined") return;
			if (event.data.size === 0) return;
			localVideoChunks.push(event.data);
		};

		setVideoChunks(localVideoChunks);
	};

	const stopRecording = () => {
		setPermission(false);
		setRecordingStatus("inactive");
		(mediaRecorder.current as any).stop();

		(mediaRecorder.current as any).onstop = () => {
			const videoBlob = new Blob(videoChunks, { type: 'video/mp4' });
			const videoUrl = URL.createObjectURL(videoBlob);
      console.log(videoUrl)
			setRecordedVideo(videoUrl);
			setVideoChunks([]);
		};
	};

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files && event.target.files[0]) {
			const videoUrl = URL.createObjectURL(event.target.files[0]);
      console.log(videoUrl)
			setRecordedVideo(videoUrl);
    }
  }

	return (
		<div>
			<h2 style={{ textAlign: 'center' }}>Engagement App</h2>
        <div>
          {
            isVideoRecord ? 
            <div className="video-controls">
              <button 
                onClick={() => setIsVideoRecord(false)} 
                type="button"
                className='video-action-btn'
              >
                Record Video
              </button>
              {
                !permission ? (
                <button 
                  onClick={getCameraPermission} 
                  type="button"
                  className='video-action-btn'
                >
                  Get Camera
                </button>
                ) : null
              }
              {
                permission && recordingStatus === "inactive" ? (
                <button 
                  onClick={startRecording}
                  type="button"
                  className='video-action-btn'
                >
                  Start Recording
                </button>
                ) : null
              }
              {
                recordingStatus === "recording" ? (
                <button 
                  onClick={stopRecording} 
                  type="button"
                  className='video-action-btn'
                >
                  Stop Recording
                </button>
                ) : null
              }
            </div> : null
          }
          {
            !isVideoRecord ?
            <div className="video-controls">
              <button 
                onClick={() => setIsVideoRecord(true)} 
                type="button"
                className='video-action-btn'
              >
                Upload Video
              </button>
              <input type="file" onChange={handleChange} style={{ color: 'transparent' }}/>
            </div> : null
          }
        </div>
      
        <div className="video-player">
          {!recordedVideo ? (
            <video ref={liveVideoFeed} autoPlay className="live-player"></video>
          ) : null}
          {recordedVideo ? (
            <div className="recorded-player">
              <video className="recorded" src={recordedVideo} controls></video>
            </div>
          ) : null}
        </div>

      {
        prediction === 1 ? 
        <div className='prediction-result'>
          Engaging
        </div> : 
        null
      }
      {
        prediction === 0 ?
        <div className='prediction-result'>
          Not Engaging
        </div> :
        null
      }
		</div>
	);
};

export default App;