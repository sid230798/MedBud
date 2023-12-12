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
          src="https://img.icons8.com/color/48/medical-doctor.png"
          alt="medical-doctor"
          className="w-8 h-8 rounded-full mr-2"
        />
      )}
      <div
        className={`flex items-center ${
          isAssistant
            ? "bg-neutral-200 text-neutral-900"
            : "bg-blue-500 text-white"
        } rounded-2xl px-3 py-2 max-w-[67%] whitespace-pre-wrap`}
        style={{ ...messageContainerStyle, overflowWrap: "anywhere" }}
      >
        {message.content}
      </div>
      {!isAssistant && (
        <img
          src="https://img.icons8.com/color/48/circled-user-male-skin-type-7--v1.png"
          alt="User DP"
          className="w-8 h-8 rounded-full ml-2"
        />
      )}
    </div>
  );
};

export default ChatMessage;
