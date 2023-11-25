import React, { useEffect, useRef, useState } from 'react';
import { Helmet } from 'react-helmet';
import Chat from '../components/Chat/Chat';
import Footer from '../components/Layout/Footer';
import Navbar from '../components/Layout/Navbar';

const Home = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleSend = async (message) => {
    const updatedMessages = [...messages, message];
    setMessages(updatedMessages);
    setLoading(true);
  
    try {
      const response = await fetch("http://127.0.0.1:5000/v1/chat/simple", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ messages: updatedMessages })
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const data = response.body;
      if (!data) {
        return;
      }
  
      const reader = data.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let isFirst = true;
  
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value);
  
        if (isFirst) {
          console.log("In isFirst");
          isFirst = false;
          setMessages((messages) => [
            ...messages,
            {
              role: "assistant",
              content: chunkValue
            }
          ]);
        } else {
          setMessages((messages) => {
            const lastMessage = messages[messages.length - 1];
            const updatedMessage = {
              ...lastMessage,
              content: lastMessage.content + chunkValue
            };
            return [...messages.slice(0, -1), updatedMessage];
          });
        }
      }
    } catch (error) {
      console.error("Error in sending message:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([
      {
        role: "assistant",
        content: `Hello! I'm MediMate, your biomedical chatbot. I provide information on health topics, assist with medical inquiries, and support healthcare needs. Need insights on medical conditions or health advice? I'm here to help. How can I assist you today?`
      }
    ]);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);


  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        content: `Hello! I'm MediMate, your biomedical chatbot. I provide information on health topics, assist with medical inquiries, and support healthcare needs. Need insights on medical conditions or health advice? I'm here to help. How can I assist you today?`
      }
    ]);
  }, []);

  return (
    <>
      <Helmet>
        <title>Medical Q&A App</title>
        {/* Other head elements */}
      </Helmet>

      <div className="flex flex-col h-screen">
        <Navbar />

        <div className="flex-1 overflow-auto sm:px-10 pb-4 sm:pb-10">
          <div className="max-w-[800px] mx-auto mt-4 sm:mt-12">
            <Chat
              messages={messages}
              loading={loading}
              onSend={handleSend}
              onReset={handleReset}
            />
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        <Footer />
      </div>
    </>
  );
};

export default Home;
