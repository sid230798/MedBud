import React from "react";

const ChatMessage = ({ message }) => {
  const isAssistant = message.role === "assistant";

  const messageContainerStyle = {
    marginLeft: isAssistant ? "10px" : "auto",
    marginRight: isAssistant ? "auto" : "10px",
  };

  return (
    <div className={`flex ${isAssistant ? "items-start" : "items-end"}`}>
      {isAssistant && (
        <img
          width="48"
          height="48"
          src="https://neopocussgrh.org/store/1/bgg.png"
          alt="medical-doctor"
          className="w-8 h-8 rounded-full mr-2"
        />
      )}
      <div
        className={`flex items-center ${
          isAssistant
            ? "bg-gradient-to-r from-pink-400 to-purple-400 text-white" // Lighter gradient for assistant
            : "bg-gradient-to-r from-pink-600 to-purple-600 text-white" // Stronger gradient for user
        } rounded-2xl px-3 py-2 max-w-[67%] whitespace-pre-wrap`}
        style={{ ...messageContainerStyle, overflowWrap: "anywhere" }}
      >
        {message.content}
      </div>
      {!isAssistant && (
        <img
          src="https://cdn-icons-png.freepik.com/512/3034/3034882.png"
          alt="User DP"
          className="w-8 h-8 rounded-full ml-2"
        />
      )}
    </div>
  );
};

export default ChatMessage;
