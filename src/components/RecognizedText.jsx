import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

function RecognizedText({ text, darkMode }) {
  const [displayText, setDisplayText] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [copied, setCopied] = useState(false)
  const textRef = useRef('')
  const indexRef = useRef(0)

  useEffect(() => {
    // Clear any existing timer
    let typingInterval;
    
    // Reset state when text changes
    setDisplayText('')
    setCopied(false)
    
    // Better handling of different response formats
    let textToDisplay = '';
    
    if (text === null || text === undefined) {
      textToDisplay = '';
    } else if (typeof text === 'object') {
      textToDisplay = text.generatedSentence || text.recognizedText || '';
    } else {
      textToDisplay = String(text); // Convert to string to handle any type
    }
    
    // Store the full text
    textRef.current = textToDisplay;
    indexRef.current = 0;
    
    // Only start animation if we have text
    if (textToDisplay) {
      setIsTyping(true);
      
      // Use a more reliable approach for typing animation
      typingInterval = setInterval(() => {
        if (indexRef.current < textRef.current.length) {
          setDisplayText(prev => textRef.current.substring(0, indexRef.current + 1));
          indexRef.current += 1;
        } else {
          clearInterval(typingInterval);
          setIsTyping(false);
        }
      }, 50);
    }
    
    // Clean up on unmount or when text changes
    return () => {
      if (typingInterval) clearInterval(typingInterval);
    };
  }, [text]);

  // Get the text to copy - handle both string and object
  const getTextToCopy = () => {
    if (typeof text === 'object' && text !== null) {
      return text.generatedSentence || text.recognizedText || '';
    }
    return text || '';
  }
  
  // Get all detected signs if available
  const getAllSigns = () => {
    if (typeof text === 'object' && text !== null && Array.isArray(text.allSigns)) {
      return text.allSigns.join(', ');
    }
    return null;
  }

  // For debugging - log the raw input
  console.log("RecognizedText received:", text);

  return (
    <motion.div 
      className="flex-1 flex flex-col"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className={`text-lg md:text-xl ${darkMode ? 'text-neutral-100' : 'text-neutral-800'}`}>
        <span className={isTyping ? 'typing-animation' : ''}>
          {displayText}
          {isTyping && <span className="cursor-blink">|</span>}
        </span>
      </div>
      
      {/* Display recognized signs if there's both generatedSentence and allSigns */}
      {typeof text === 'object' && text !== null && 
       text.generatedSentence && text.allSigns && (
        <div className={`mt-3 text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
          <span className="font-medium">Recognized signs:</span> {getAllSigns()}
        </div>
      )}
      
      {/* Display just the recognized text if there's only that */}
      {typeof text === 'object' && text !== null && 
       text.recognizedText && !text.generatedSentence && !text.allSigns && (
        <div className={`mt-3 text-sm ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
          <span className="font-medium">Recognized sign:</span> {text.recognizedText}
        </div>
      )}
      
      <div className="mt-auto flex justify-between items-center pt-4">
        <div className={`text-xs ${darkMode ? 'text-neutral-400' : 'text-neutral-500'}`}>
          {new Date().toLocaleTimeString()}
        </div>
        
        <div className="relative">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`p-1.5 rounded ${
              darkMode ? 'hover:bg-[#3c4043]/40' : 'hover:bg-[#f1f3f4]'
            } transition-colors duration-200`}
            onClick={() => {
              navigator.clipboard.writeText(getTextToCopy());
              setCopied(true);
              setTimeout(() => setCopied(false), 2000);
            }}
            title="Copy to clipboard"
          >
            {!copied ? (
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 ${darkMode ? 'text-[#9aa0a6]' : 'text-[#5f6368]'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 ${darkMode ? 'text-[#8ab4f8]' : 'text-[#1a73e8]'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            )}
          </motion.button>
          
          <AnimatePresence>
            {copied && (
              <motion.div 
                className={`absolute bottom-full right-0 mb-1 px-2 py-1 text-xs rounded ${darkMode ? 'bg-[#3c4043] text-[#e8eaed]' : 'bg-[#f1f3f4] text-[#202124]'}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
              >
                Copied!
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
      
      {/* Add this CSS for the blinking cursor effect */}
      <style jsx>{`
        .cursor-blink {
          animation: blink 1s step-end infinite;
        }
        
        @keyframes blink {
          from, to { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </motion.div>
  );
}

export default RecognizedText;