import React, { useEffect, useState } from "react";

const FileUploader = ({ onFileUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isFileUploaded, setIsFileUploaded] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) {
      alert("Please select a text file to upload.");
      return;
    }

    setSelectedFile(file);
  };

  useEffect(() => {
    if (selectedFile) {
      const reader = new FileReader();

      reader.onload = (event) => {
        const fileContent = event.target.result;
        console.log(fileContent);
        onFileUpload(fileContent);
        setIsFileUploaded(true);
        alert("File uploaded successfully.");
      };

      reader.readAsText(selectedFile);
    }
  }, [selectedFile]);

  return (
      <div className="flex items-center">
        <label
            htmlFor="file-input"
            className={`cursor-pointer px-1 py-2 rounded-md mr-2 text-white ${
                isFileUploaded
                    ? "bg-green-500 hover:bg-green-600"
                    : "bg-blue-500 hover:bg-blue-600"
            }`}
        >
          <img
              width="20"
              height="20"
              src={
                !isFileUploaded
                    ? "https://static.vecteezy.com/system/resources/previews/015/337/675/non_2x/transparent-upload-icon-free-png.png" // New icon URL
                    : "https://img.icons8.com/ios-filled/50/document--v1.png" // Icon for uploaded file
              }
              alt="upload-icon"
          />
        </label>
        <input
            disabled={isFileUploaded}
            type="file"
            accept=".txt"
            onChange={handleFileChange}
            className="hidden"
            id="file-input"
        />
      </div>
  );
};

export default FileUploader;