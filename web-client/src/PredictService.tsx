export class PredictService {

    predictVideo = async (videoUri: string) => {

        const fetchedResponse = await fetch(videoUri);
        const videoBlob = await fetchedResponse.blob();

        const myFile = new File(
            [videoBlob],
            "video.mp4",
            { type: 'video/mp4' }
        );
        
        const formData = new FormData()
        formData.append('file', myFile, 'video.mp4');
    
        const requestOptions = {
          method: 'POST',
          body: formData,
        };
        const response = await fetch('http://localhost:8000/predict', requestOptions)
        return response

    }

}