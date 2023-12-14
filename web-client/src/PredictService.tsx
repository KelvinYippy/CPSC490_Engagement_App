export class PredictService {

    predictVideo = async (videoUri: string) => {
        // 8081
        console.log("hi there!")
        const fetchedResponse = await fetch(videoUri);
        const videoBlob = await fetchedResponse.blob();
        console.log("hi there!")
        // Create a FormData object and append the video file

        const myFile = new File(
            [videoBlob],
            "video.mp4",
            { type: 'video/mp4' }
        );
        
        const formData = new FormData()
        // formData.append('video', {
        //   uri: videoUri, // Replace with your local video URI
        //   name: 'video.mp4', // Replace with desired file name
        //   type: 'video/mp4', // Replace with the correct MIME type if necessary
        // });
    
        // Append the video blob
        formData.append('file', myFile, 'video.mp4');
    
        console.log(formData.get('file'))

        // Set up fetch options
        const requestOptions = {
          method: 'POST',
          body: formData,
        //   headers: {
        //     Accept: 'application/json',
        //   },
        };
    
        // Make the POST request to your FastAPI endpoint
        // const apiUrl = 'http://your-fastapi-endpoint/upload';
        // const uploadResponse = await fetch(apiUrl, requestOptions);
        // Make the POST request to your FastAPI endpoint
        console.log("going to here!")
        const response = await fetch('http://localhost:8000/predict', requestOptions)
        return response
    }

    // private predictVideo = async (videoUri: string) => {
    //     const 
    //     const teams: Team[] = await (await fetch(`http://localhost:8081/${team_type}`)).json()
    //     return teams
    // }

    // fetchTrendingTeams = async () => await this.fetchTeamHelper("teams")

    // fetchClubTeams = async () => await this.fetchTeamHelper("clubs")

    // fetchNationalTeams = async () => await this.fetchTeamHelper("national_teams")

}