import { motion } from 'framer-motion'

function Footer({ darkMode }) {
  return (
    <motion.footer 
      className={`py-5 border-t ${darkMode ? 'border-[#3c4043]/30 text-[#9aa0a6]' : 'border-[#dadce0] text-[#5f6368]'}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <div className="container mx-auto px-6 text-center">
        <p className="text-xs">
          <span className={`${darkMode ? 'text-[#8ab4f8]' : 'text-[#1a73e8]'}`}>Gestura</span> | American Sign Language Recognition
        </p>
        <p className="text-xs mt-1">
          &copy; {new Date().getFullYear()} Gestura. All rights reserved.
        </p>
      </div>
    </motion.footer>
  )
}

export default Footer