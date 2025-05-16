import { createContext, useState } from 'react';

export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  
  return (
    <AppContext.Provider value={{ 
      isLoading, 
      setIsLoading, 
      recognizedText, 
      setRecognizedText 
    }}>
      {children}
    </AppContext.Provider>
  );
};