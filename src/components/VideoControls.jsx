import { motion } from 'framer-motion'

function VideoControls({ darkMode, isRecording, hasCamera, onEnableCamera, onStartRecording, onStopRecording }) {
  return (
    <div className="flex justify-center space-x-3">
      {!hasCamera ? (
        <motion.button
          onClick={onEnableCamera}
          className={`px-5 py-2.5 rounded-lg flex items-center ${
            darkMode 
              ? 'bg-[#8ab4f8] hover:bg-[#8ab4f8]/90 text-[#202124]' 
              : 'bg-[#1a73e8] hover:bg-[#1a73e8]/90 text-white'
          } transition-colors duration-200`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          <span className="font-medium">Enable camera</span>
        </motion.button>
      ) : !isRecording ? (
        <motion.button
          onClick={onStartRecording}
          className={`px-5 py-2.5 rounded-lg flex items-center ${
            darkMode 
              ? 'bg-[#34a853] hover:bg-[#34a853]/90 text-white' 
              : 'bg-[#34a853] hover:bg-[#34a853]/90 text-white'
          } transition-colors duration-200`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="font-medium">Start recording</span>
        </motion.button>
      ) : (
        <motion.button
          onClick={onStopRecording}
          className={`px-5 py-2.5 rounded-lg flex items-center ${
            darkMode 
              ? 'bg-[#f28b82] hover:bg-[#f28b82]/90 text-[#202124]' 
              : 'bg-[#ea4335] hover:bg-[#ea4335]/90 text-white'
          } transition-colors duration-200`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
          </svg>
          <span className="font-medium">Stop recording</span>
        </motion.button>
      )}
    </div>
  )
}

export default VideoControls
