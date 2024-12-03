import React, {useEffect, useRef, useState} from 'react';
import {Helmet} from 'react-helmet';
import Chat from '../components/Chat/Chat';
import Footer from '../components/Layout/Footer';
import Navbar from '../components/Layout/Navbar';
import {API_BASE_URL, API_ENDPOINTS, MODELS_AVAILABLE} from './APIConfig';
import logo from '../store/logo.png'

const Home = () => {
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState('Debug');
    const [selectedDataset, setSelectedDataset] = useState('nfcorpus');
    const [isRAGEnabled, setIsRAGEnabled] = useState(false);
    const [medicalHistory, setMedicalHistory] = useState('');
    const messagesEndRef = useRef(null);
    const getApiEndpoint = () => {
        if (selectedModel === "Debug")
            return API_ENDPOINTS.SIMPLE_CHAT;
        if (isRAGEnabled)
            return API_ENDPOINTS.RAG_CHAT;
        else
            return API_ENDPOINTS.OPENAI_CHAT;
    };

    const getSelectedModel = () => {
        switch (selectedModel) {
            case "OpenAI gpt-4.0-turbo":
                return MODELS_AVAILABLE.GPT_3_5;
            case "OpenAI da-vinci-003":
                return MODELS_AVAILABLE.DA_VINCI;
            default:
                return MODELS_AVAILABLE.GPT_3_5;
        }
    }

    const scrollToBottom = () => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({behavior: "smooth"});
        }
    };

    const handleSend = async (message) => {
        const updatedMessages = [...messages, message];
        setMessages(updatedMessages);
        setLoading(true);

        try {
            const endpoint = getApiEndpoint();
            const model = getSelectedModel(selectedModel);
            const dataset = selectedDataset;
            const bodyPayload = {
                messages: updatedMessages,
                medicalHistory: medicalHistory
            };
            const response = await fetch(`${API_BASE_URL}${endpoint}?model=${encodeURIComponent(model)}&dataset=${encodeURIComponent(dataset)}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(bodyPayload)
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
                const {value, done: doneReading} = await reader.read();
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
                content: `Hello! I'm MedBud, your biomedical chatbot. I provide information on health topics, assist with medical inquiries, and support healthcare needs. Need insights on medical conditions or health advice? I'm here to help. How can I assist you today?`
            }
        ]);
    };

    const handleModelChange = (model) => {
        setSelectedModel(model);
        setIsRAGEnabled(false);
        handleDatasetChange('nfcorpus');
    };

    const handleDatasetChange = (dataset) => {
        setSelectedDataset(dataset);
    };

    const handleRAGToggle = () => {
        setIsRAGEnabled(!isRAGEnabled);
        setSelectedDataset('nfcorpus');
    };
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        setMessages([
            {
                role: "assistant",
                content: `Hello! I'm MedBud, your biomedical chatbot. I provide information on health topics, assist with medical inquiries, and support healthcare needs. Need insights on medical conditions or health advice? I'm here to help. How can I assist you today?`
            }
        ]);
    }, []);

    return (
        <>
            <Helmet>
                <title>MedBud - Your personal Medical QnA assistant!</title>
                <link rel="icon" href={logo} type="image/png"/>
            </Helmet>
            {/* style={pageStyle} */}
            <div className="flex flex-col h-screen">
                <Navbar/>

                <div className="flex-1 overflow-auto sm:px-10 pb-4 sm:pb-10">
                    <div className="max-w-[800px] mx-auto mt-4 sm:mt-12">
                        <Chat
                            messages={messages}
                            loading={loading}
                            onSend={handleSend}
                            onReset={handleReset}
                            onModelChange={handleModelChange}
                            onDatasetChange={handleDatasetChange}
                            isRAGEnabled={isRAGEnabled}
                            handleRAGToggle={handleRAGToggle}
                            setMedicalHistory={setMedicalHistory}
                        />
                        <div ref={messagesEndRef}/>
                    </div>
                </div>

                <Footer/>
            </div>
        </>
    );
};

export default Home;
