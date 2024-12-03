// src/components/ModelSelection/RAGToggle.js

import React from "react";

const RAGToggle = ({ isRAGEnabled, handleRAGToggle, handleReset }) => {
  const handleRAGChange = () => {
    handleRAGToggle();
    handleReset();
  };
  return (
    <label className="flex items-center space-x-3 mt-1">
      <input
        type="checkbox"
        className="form-tick h-6 w-6 border border-gray-300 rounded-md checked:bg-blue-600 checked:border-transparent focus:outline-none"
        checked={isRAGEnabled}
        onChange={handleRAGChange}
        onReset={handleReset}
      />
      <span className="text-gray-700 dark:text-black font-normal">
        <b>Enable RAG + CoT</b>
      </span>
    </label>
  );
};

export default RAGToggle;
