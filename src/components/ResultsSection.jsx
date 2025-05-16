import { useContext } from 'react'
import { motion } from 'framer-motion'
import RecognizedText from './RecognizedText'
import { AppContext } from '../context/AppContext'

function ResultsSection({ darkMode }) {
  const { isLoading, recognizedText } = useContext(AppContext)

  return (
    <motion.div 
      className={`w-full md:w-1/2 rounded-xl overflow-hidden ${
        darkMode ? 'bg-[#2d2d2d]' : 'bg-white'
      } ${darkMode ? 'border border-[#3c4043]/30' : 'border border-[#dadce0]'} shadow-sm`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className={`text-lg font-medium ${darkMode ? 'text-[#e8eaed]' : 'text-[#202124]'}`}>
            Recognition Results
          </h2>
          <div className={`text-xs font-medium px-2 py-1 rounded-full ${isLoading ? 'bg-[#fbbc04]/10 text-[#fbbc04]' : recognizedText ? 'bg-[#34a853]/10 text-[#34a853]' : darkMode ? 'bg-[#3c4043]/30 text-[#9aa0a6]' : 'bg-[#f1f3f4] text-[#5f6368]'}`}>
            {isLoading ? 'Processing' : recognizedText ? 'Completed' : 'Waiting'}
          </div>
        </div>
        
        <div className={`min-h-[250px] p-4 rounded-lg ${
          darkMode ? 'bg-[#202124] border border-[#3c4043]/50' : 'bg-[#f8f9fa] border border-[#dadce0]'
        } flex flex-col`}>
          {isLoading ? (
            <div className="flex-1 flex flex-col items-center justify-center">
              <div className="w-10 h-10 border-3 border-[#fbbc04] border-t-transparent rounded-full animate-spin mb-3"></div>
              <p className={`text-sm ${darkMode ? 'text-[#9aa0a6]' : 'text-[#5f6368]'}`}>
                Processing your sign language...
              </p>
            </div>
          ) : recognizedText ? (
            <RecognizedText text={recognizedText} darkMode={darkMode} />
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                className={`h-10 w-10 mx-auto mb-3 ${darkMode ? 'text-[#9aa0a6]' : 'text-[#5f6368]'}`} 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className={`text-sm ${darkMode ? 'text-[#9aa0a6]' : 'text-[#5f6368]'}`}>
                Record a video using ASL signs to see the recognized text here.
              </p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export default ResultsSection