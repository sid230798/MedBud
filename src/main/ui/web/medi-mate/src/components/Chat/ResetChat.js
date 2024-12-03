import React from "react";

const ResetChat = ({ onReset }) => {
  return (
    <div className="flex flex-row items-center">
      <button
        className="text-white font-semibold rounded-lg px-4 py-2 bg-purple-500 hover:bg-purple-600 focus:outline-none focus:ring-1 focus:ring-purple-500"
        onClick={() => onReset()}
      >
        Reset Chat
      </button>
    </div>
  );
};
export default ResetChat;
