import React, { useState } from 'react';
import FileUploader from './FileUploader';

const ChooseModels = ({ selectedMethod, setSelectedIcon }) => {
  const [selectedIcon, setSelectedModel] = useState(null);

  const handleIconClick = (icon) => {
    setSelectedModel((prevModel) => (prevModel === icon ? null : icon));
    setSelectedIcon((prevIcon) => (prevIcon === icon ? null : icon));
  };

  if (!['pipeline', 'multitask', 'end2end', 'instruction'].includes(selectedMethod)) {
    return null;
  }

  return (
    <div>
      <div className='py-10 items-center flex justify-center'>
        <p className='text-xl underline'> Select your models: </p>
      </div>

      {selectedMethod === 'pipeline' && (
        <div className='overflow-auto mx-auto grid-rows-2'>
         <ul className="inline-grid items-center mx-auto" style={{ gridTemplateColumns: 'repeat(2, minmax(12rem, 1fr))' , gridTemplateRows: 'repeat(1, minmax(7rem, 4fr))' }}>            
          <li>
            <button
              type="button"
              className="group text-sm font-semibold w-full flex flex-col items-center "
              onClick={() => handleIconClick('pipelinevit5')}
              style={{
                color: selectedIcon === 'pipelinevit5' ? 'orange' : 'initial',
                transform: selectedIcon === 'pipelinevit5' ? 'scale(1.1)' : 'scale(1)',
                transition: 'transform 0.3s ease',
              }}
            >
              <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
                <image href="./src/Figs/Models/1.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Pipeline ViT5
            </button>
          </li>
          <li>
            <button
              type="button"
              className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedIcon === 'instruction' ? 'selected' : ''}`}
              onClick={() => handleIconClick('pipelinebartpho')}
              style={{
                color: selectedIcon === 'pipelinebartpho' ? 'cyan' : 'initial',
                transform: selectedIcon === 'pipelinebartpho' ? 'scale(1.1)' : 'scale(1)',
                transition: 'transform 0.3s ease',
              }}
            >
              <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-indigo-500 dark:text-indigo-400">
                <image href="./src/Figs/Models/2.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
              </svg>
              Pipeline BARTPho
            </button>
          </li>
         </ul>
        </div>
      )}

      {selectedMethod === 'multitask' && (
        <div className='overflow-auto mx-auto grid-rows-2'>
        <ul className="inline-grid items-center mx-auto" style={{ gridTemplateColumns: 'repeat(2, minmax(12rem, 1fr))' , gridTemplateRows: 'repeat(1, minmax(7rem, 4fr))' }}>            
         <li>
           <button
             type="button"
             className="group text-sm font-semibold w-full flex flex-col items-center "
             onClick={() => handleIconClick('multitaskvit5')}
             style={{
               color: selectedIcon === 'multitaskvit5' ? 'brown' : 'initial',
               transform: selectedIcon === 'multitaskvit5' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
               <image href="./src/Figs/Models/3.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
             </svg>
             Multitask ViT5
           </button>
         </li>
         <li>
           <button
             type="button"
             className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedIcon === 'instruction' ? 'selected' : ''}`}
             onClick={() => handleIconClick('multitaskbartpho')}
             style={{
               color: selectedIcon === 'multitaskbartpho' ? 'teal' : 'initial',
               transform: selectedIcon === 'multitaskbartpho' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-indigo-500 dark:text-indigo-400">
               <image href="./src/Figs/Models/4.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
             </svg>
             Multitask BARTPho
           </button>
         </li>
        </ul>
       </div>
      )}

      {selectedMethod === 'end2end' && (
        <div className='overflow-auto mx-auto grid-rows-2'>
        <ul className="inline-grid items-center mx-auto" style={{ gridTemplateColumns: 'repeat(2, minmax(12rem, 1fr))' , gridTemplateRows: 'repeat(1, minmax(7rem, 4fr))' }}>            
         <li>
           <button
             type="button"
             className="group text-sm font-semibold w-full flex flex-col items-center "
             onClick={() => handleIconClick('end2endvit5')}
             style={{
               color: selectedIcon === 'end2endvit5' ? 'maroon' : 'initial',
               transform: selectedIcon === 'end2endvit5' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
               <image href="./src/Figs/Models/5.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
             </svg>
             End2End ViT5
           </button>
         </li>
         <li>
           <button
             type="button"
             className={`group text-sm font-semibold w-full flex flex-col items-center ${selectedIcon === 'instruction' ? 'selected' : ''}`}
             onClick={() => handleIconClick('end2endbartpho')}
             style={{
               color: selectedIcon === 'end2endbartpho' ? 'navy' : 'initial',
               transform: selectedIcon === 'end2endbartpho' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-indigo-500 dark:text-indigo-400">
               <image href="./src/Figs/Models/6.png" width="50" height="50" fill="currentColor" fillOpacity="0" stroke="currentColor" strokeWidth="2"/>
             </svg>
             End2End BARTPho
           </button>
         </li>
        </ul>
       </div>
      )}

      {selectedMethod === 'instruction' && (
        <div className='overflow-auto mx-auto grid-rows-2'>
        <ul className="inline-grid items-center mx-auto" style={{ gridTemplateColumns: 'repeat(4, minmax(12rem, 1fr))' , gridTemplateRows: 'repeat(1, minmax(7rem, 4fr))' }}>            
         <li>
           <button
             type="button"
             className="group text-sm font-semibold w-full flex flex-col items-center "
             onClick={() => handleIconClick('invit5')}
             style={{
               color: selectedIcon === 'invit5' ? 'olive' : 'initial',
               transform: selectedIcon === 'invit5' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
               <image href="./src/Figs/Models/7.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
             </svg>
             Inst ViT5
           </button>
         </li>
         <li>
           <button
             type="button"
             className="group text-sm font-semibold w-full flex flex-col items-center "
             onClick={() => handleIconClick('inbartpho')}
             style={{
               color: selectedIcon === 'inbartpho' ? 'salmon' : 'initial',
               transform: selectedIcon === 'inbartpho' ? 'scale(1.1)' : 'scale(1)',
               transition: 'transform 0.3s ease',
             }}
           >
             <svg width="50" height="50" fill="none" aria-hidden="true" className="mb-6 text-slate-300 group-hover:text-slate-400 dark:text-slate-600 dark:group-hover:text-slate-500">
               <image href="./src/Figs/Models/8.png" width="50" height="50" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
             </svg>
             Inst BARTPho
           </button>
         </li>
        </ul>
       </div>
      )}
      {selectedIcon && <FileUploader selectedModel={selectedIcon} />}
    </div>
  );
};

export default ChooseModels;