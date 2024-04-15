import React, { useState } from 'react';
import ChooseModels from './ChooseModels';
import FileUploader from './FileUploader';

const ChooseMethods = () => {
  const [selectedMethod, setSelectedMethod] = useState(null);

  const handleIconClick = (method) => {
    setSelectedMethod((prevMethod) => (prevMethod === method ? null : method));
  };

  return (
    <div className='overflow-auto mx-auto grid-rows-2'>
      <ul className="inline-grid items-center pt-2 mx-auto" style={{ gridTemplateColumns: 'repeat(4, minmax(12rem, 1fr))' , gridTemplateRows: 'repeat(1, minmax(7rem, 4fr))' }}>
        <li>
          <button
            type="button"
            className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedMethod === 'pipeline' ? 'selected' : ''}`}
            onClick={() => handleIconClick('pipeline')}
            style={{
              color: selectedMethod === 'pipeline' ? 'indigo' : 'initial',
              transform: selectedMethod === 'pipeline' ? 'scale(1.1)' : 'scale(1)',
              transition: 'transform 0.3s ease',
            }}
          >
            <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-indigo-500 dark:text-indigo-400">
              <image href="./src/Figs/Methods/pipeline_icon.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Pipeline
          </button>
        </li>
        <li>
          <button
            type="button"
            className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedMethod === 'multitask' ? 'selected' : ''}`}
            onClick={() => handleIconClick('multitask')}
            style={{
              color: selectedMethod === 'multitask' ? 'green' : 'initial',
              transform: selectedMethod === 'multitask' ? 'scale(1.1)' : 'scale(1)',
              transition: 'transform 0.3s ease',
            }}
          >
            <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
              <image href="./src/Figs/Methods/multitask_icon.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Multitask
          </button>
        </li>
        <li>
          <button
            type="button"
            className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedMethod === 'end2end' ? 'selected' : ''}`}
            onClick={() => handleIconClick('end2end')}
            style={{
              color: selectedMethod === 'end2end' ? 'gold' : 'initial',
              transform: selectedMethod === 'end2end' ? 'scale(1.1)' : 'scale(1)',
              transition: 'transform 0.3s ease',
            }}
          >
            <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
              <image href="./src/Figs/Methods/end2end_icon.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            End2End
          </button>
        </li>
        <li>
          <button
            type="button"
            className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedMethod === 'instruction' ? 'selected' : ''}`}
            onClick={() => handleIconClick('instruction')}
            style={{
              color: selectedMethod === 'instruction' ? 'darkslategray' : 'initial',
              transform: selectedMethod === 'instruction' ? 'scale(1.1)' : 'scale(1)',
              transition: 'transform 0.3s ease',
            }}
          >
            <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-indigo-500 dark:text-indigo-400">
              <image href="./src/Figs/Methods/instruction_icon.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
            </svg>
            Instruction
          </button>
        </li>
      </ul>
      <ChooseModels selectedMethod={selectedMethod} setSelectedMethod={setSelectedMethod} />
    </div>
  );
};

export default ChooseMethods;