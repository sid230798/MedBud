import React, { useEffect, useRef, useState } from "react";
import { IconArrowUp } from "@tabler/icons-react";
import MicButton from "../SpeechRecognition/MicButton";
import FileUploader from "../UploadContext/FileUploader";

const ChatInput = ({ onSend, setMedicalHistory }) => {
  const [content, setContent] = useState("");
  const [placeholder, setPlaceholder] = useState("Type to talk to MedBud...");
  const [messageSent, setMessageSent] = useState(false);
  const textareaRef = useRef(null);

  const handleFileUpload = (fileContent) => {
    console.log("Uploaded file content:", fileContent);
    setMedicalHistory(fileContent);
  };

  const handleChange = (e) => {
    const value = e.target.value;
    if (value.length > 4000) {
      alert("Message limit is 4000 characters");
      return;
    }

    setContent(value);
  };

  const handleTranscription = (transcribedText) => {
    setContent(transcribedText);
    if (!transcribedText) setPlaceholder("Type to talk to Medbud...");
  };

  const updatePlaceholder = (isListening) => {
    setPlaceholder(
        isListening ? "Speak to Chat with MedBud..." : "Type to talk to MedBud...",
    );
  };

  const handleSend = () => {
    if (!content) {
      alert("Please enter a message");
      return;
    }
    onSend({ role: "user", content });
    setContent("");
    setMessageSent(true);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "inherit";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [content]);

  return (
      <div className="relative flex">
        <FileUploader onFileUpload={handleFileUpload} />
        <textarea
            ref={textareaRef}
            className="min-h-[44px] rounded-lg pl-4 pr-12 py-2 focus:outline-none focus:ring-1 focus:ring-neutral-300 border-2 border-neutral-200"
            style={{ resize: "none", width: "85%" }}
            placeholder={placeholder}
            value={content}
            rows={1}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
        />
        <button onClick={handleSend}>
          <IconArrowUp className="absolute right-2 bottom-1.5 h-8 w-8 hover:cursor-pointer rounded-full p-1 bg-blue-500 text-white hover:opacity-80" />
        </button>

        <MicButton
            onTranscription={handleTranscription}
            updatePlaceholder={updatePlaceholder}
            messageSent={messageSent}
            setMessageSent={setMessageSent}
        />
      </div>
  );
};

export default ChatInput;