import React from 'react';

const SendData2API = () => {
  const [fileContent, setFileContent] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];

    if (file) {
      const reader = new FileReader();

      reader.onload = (e) => {
        const content = e.target.result;
        setFileContent({ file, content });
      };

      reader.readAsText(file);
    }
  }, []);

  const [apiResponse, setApiResponse] = useState(null);

  const handleSendData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/gen', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: fileContent.content }),
      });
  
      const result = await response.json();
      console.log('API response:', result);

      // Update state with the API response
      setApiResponse(result.lowercase_data);
    } catch (error) {
      console.error('Error sending data to API:', error);
    }
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    noClick: true, // Prevents opening the file dialog on click
  });

  const handleTextAreaChange = (event) => {
    const content = event.target.value;
    setFileContent({ content });
  };

  const handleFileDelete = () => {
    setFileContent(null);
  };
  return (
    <div>
      {fileContent && (
      <div className="relative mb-3 flex md:auto" data-te-input-wrapper-init>
        <textarea
          className="peer block min-h-[auto] w-full rounded border border-black bg-transparent px-3 py-[0.32rem] leading-[1.6] outline-none transition-all duration-200 ease-linear focus:placeholder:opacity-100 data-[te-input-state-active]:placeholder:opacity-100 motion-reduce:transition-none dark:text-neutral-200 dark:placeholder:text-neutral-200 [&:not([data-te-input-placeholder-active])]:placeholder:opacity-0"
          id="exampleFormControlTextarea1"
          rows="3"
          placeholder="Your message"
        ></textarea>
        <label
          htmlFor="exampleFormControlTextarea1"
          className="pointer-events-none absolute left-3 top-1 mb-1 max-w-[90%] origin-[0_0] truncate pt-[0.37rem] leading-[1.6] text-neutral-500 transition-all duration-200 ease-out peer-focus:-translate-y-[0.9rem] peer-focus:scale-[0.8] peer-focus:text-primary peer-data-[te-input-state-active]:-translate-y-[0.9rem] peer-data-[te-input-state-active]:scale-[0.8] motion-reduce:transition-none dark:text-neutral-200 dark:peer-focus:text-primary"
        >
          Example
        </label>
      </div>
    )};
    <div className="mt-4 flex pt-10 items-center justify-center mx-auto col-start-1 row-start-3 self-center sm:mt-0 sm:col-start-2 sm:row-start-2 sm:row-span-2 lg:mt-6 lg:col-start-1 lg:row-start-3 lg:row-end-4">
      <button
        type="button"
        className="bg-indigo-600 hover:bg-blue-400 text-white text-sm leading-6 font-medium py-2 px-3 rounded-lg"
        >
        Generation
      </button>
    </div>
    </div>
  );
};

export default SendData2API;