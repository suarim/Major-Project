import { useState, useRef, useContext } from 'react';
import { motion } from 'framer-motion';
import VideoControls from './VideoControls';
import { AppContext } from '../context/AppContext';

function VideoSection({ darkMode }) {
  const { setRecognizedText, setIsLoading } = useContext(AppContext);
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recordedChunks, setRecordedChunks] = useState([]);
  const videoRef = useRef(null);

  const enableCamera = async () => {
    try {
      const videoStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      videoRef.current.srcObject = videoStream;
      await videoRef.current.play();
      setStream(videoStream);
    } catch (error) {
      console.error('Failed to enable camera:', error);
      alert('Could not access camera. Please allow permissions.');
    }
  };

  const startRecording = () => {
    if (!stream) {
      alert('Camera is not enabled.');
      return;
    }

    // Clear previous recordings
    setRecordedChunks([]);
    
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    setMediaRecorder(recorder);

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        setRecordedChunks((prev) => [...prev, event.data]);
      }
    };

    recorder.start(1000); // Capture chunks every second
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    setIsRecording(false);
    
    // Process video after a short delay to ensure all chunks are collected
    setTimeout(() => processRecording(), 500);
  };

  const processRecording = async () => {
    if (recordedChunks.length === 0) {
      alert('No video data recorded.');
      return;
    }

    try {
      setIsLoading(true);
      
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const formData = new FormData();
      formData.append('video', blob, 'recording.webm');

      // Send to backend server
      const response = await fetch('http://localhost:5000/api/process-video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process video');
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      // Handle the response
      if (data.generatedSentence) {
        setRecognizedText(data); // Pass the whole object
      } else if (data.recognizedText) {
        setRecognizedText(data.recognizedText);
      } else {
        setRecognizedText('No signs recognized.');
      }
    } catch (error) {
      console.error('Error processing video:', error);
      setRecognizedText('Error processing video: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      className={`w-full md:w-1/2 rounded-xl overflow-hidden ${darkMode ? 'bg-[#2d2d2d]' : 'bg-white'} ${darkMode ? 'border border-[#3c4043]/30' : 'border border-[#dadce0]'} shadow-sm`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="p-5">
        <div className="relative aspect-video rounded-lg overflow-hidden">
          <video ref={videoRef} muted playsInline className="w-full h-full object-cover" />
        </div>
        <VideoControls
          darkMode={darkMode}
          isRecording={isRecording}
          hasCamera={!!stream}
          onEnableCamera={enableCamera}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
        />
      </div>
    </motion.div>
  );
}

export default VideoSection;