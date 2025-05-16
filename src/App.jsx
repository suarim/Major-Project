import { useState } from 'react'
import VideoSection from './components/VideoSection'
import ResultsSection from './components/ResultsSection'
import Header from './components/Header'
import { AppProvider } from './context/AppContext'
import Footer from './components/Footer'

function App() {
  const [darkMode, setDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  )

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

  return (
    <AppProvider>
      <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-[#1f1f1f] text-white' : 'bg-[#f8f9fa] text-[#202124]'}`}>
        <Header darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
        
        <main className="container mx-auto px-4 py-6 md:py-8 flex flex-col items-center">
          <div className="w-full max-w-6xl flex flex-col md:flex-row gap-8 md:gap-10">
            <VideoSection darkMode={darkMode} />
            <ResultsSection darkMode={darkMode} />
          </div>
        </main>
        
        <Footer darkMode={darkMode} />
      </div>
    </AppProvider>
  )
}

export default App