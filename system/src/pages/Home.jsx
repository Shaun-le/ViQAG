import React, { useState } from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGithub } from '@fortawesome/free-brands-svg-icons';
import { MathJax, MathJaxContext} from 'better-react-mathjax';
import ChooseMethos from '../component/ChooseMethos';

const Home = () => {
  const gradientColors = 'radial-gradient(circle at 22% 43%, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 1) 99%)';
  const Q = <MathJaxContext><MathJax inline>{' \\(Q\\)'}</MathJax></MathJaxContext>
  const C = <MathJaxContext><MathJax inline>{' \\(C\\)'}</MathJax></MathJaxContext>
  const f = <MathJaxContext><MathJax inline>{' \\(f()\\)'}</MathJax></MathJaxContext>
  const theta = <MathJaxContext><MathJax inline>{' \\(\\theta \\)'}</MathJax></MathJaxContext>
  const [toggle, setToggle] = useState(false);
  return (
    <section
      className='w-full h-screen relative'
      style={{
        background: gradientColors,
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}
    >
      {/* <section 1*/}
      <section className='grid lg:grid-cols-2 place-items-center pb-8 md:pt-32 md:pb-18'>
        <div>
          <h1 className="text-center text-black lg:text-6xl xl:text-6xl font-bold lg:tracking-tight xl:tracking-tighter">
            Vietnamese Question and Answer Generation
          </h1>
          <p className="pt-6 md:p-8 md:text-left space-y-4">
              Question and answer generation (QAG) is a natural language processing (NLP) task 
              that generates a question and an answer at the same time by using context information. 
              The input context can be represented in the form of structured information in a database or 
              raw text from news articles. The outputs of QAG systems can be directly applied to 
              several NLP applications such as question answering and question-answer pair data augmentation.
          </p>
          <div className="text-center flex-col sm:flex-row">
            <a
                href=""
                className="bg-gray-900 text-white rounded-md px-3 py-3 text-lg font-medium mb-2 sm:mb-0 sm:mr-3"
                rel="noopener noreferrer"
            >
                Paper
            </a>
            <a
              size="lg"
              style={{ outline: 'none' }}
              rel="noopener noreferrer"
              href="https://github.com/Shaun-le/ViQAG.git"
              className="bg-white text-black rounded-md px-3 py-3 text-lg font-medium border border-black"
              target="_blank"
            >
              <FontAwesomeIcon icon={faGithub} className="text-black w-4 h-4" />
              GitHub Repo
            </a>
          </div>
        </div>
        <div className="md:order-1 hidden md:block">
          <img src="./src/Figs/QAG/CoRwH.png" alt="Converse of Robot with Human" width="650" className="flex-none rounded-lg bg-slate-100" loading="lazy" />
          <div className="text-right mt-2 text-sm text-gray-500">Illustration: Shaun Le</div>
        </div>
      </section>

      {/* <section 2*/}
      <div className="xs:mt-96 md:mt-112 lg:mt-128 w-full border-t border-black pt-8"></div>
      <section className='h-screen'>
        <div className="container py-20 2xl:px-60 d-md-flex flex-column justify-content-md-center align-items-md-start">
          <h1 id="implementation-steps" className='text-black lg:text-6xl xl:text-4xl'>Rock'n roll</h1>
          <div className='video-container py-10'>
            <iframe 
              width="800" 
              height="470" 
              src="https://www.youtube.com/embed/hIlQgg7ygQU?si=2Yxs2SzyS5WC6xPI" 
              title="YouTube video player" 
              frameBorder="0" 
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
              allowFullScreen
            ></iframe>
          </div>


          <div className="container mx-auto py-10 2xl:px-150 d-md-flex flex-column justify-content-md-center align-items-md-start">
            {/*<FileUploader />*/}
            <div className='pb-10 items-center flex justify-center'>
              <p className='text-xl underline'> Select your methods: </p>
            </div>
            <ChooseMethos />
          </div>
        </div>
      </section>
      <div className='pt-96'>
      
      <footer class="bg-white dark:bg-gray-900 pt-96">
          <div class="mx-auto w-full max-w-screen-xl p-4 py-6 lg:py-8">
              <div class="md:flex md:justify-between">
                  <div class="grid grid-cols-2 gap-8 sm:gap-6 sm:grid-cols-3">            
                  </div>
              </div>
              <hr class="my-6 border-gray-200 sm:mx-auto dark:border-gray-700 lg:my-8" />
              <div class="sm:flex sm:items-center sm:justify-between">
                  <span class="text-sm text-gray-500 sm:text-center dark:text-gray-400">© 2022 <a href=" " class="hover:underline">Alpha-Lab™</a>
                  </span>
                  <div class="flex mt-4 sm:justify-center sm:mt-0">
                      <a href="mailto:alpha.utehy@gmail.com" class="text-gray-500 hover:text-gray-900 dark:hover:text-white ms-5">
                          <img src="./src/Figs/Logo/mail.png" class="w-4 h-4" aria-hidden="true" alt="Email Icon"/>
                          <span class="sr-only">Gmail</span>
                      </a>
                      <a href="https://github.com/Shaun-le" class="text-gray-500 hover:text-gray-900 dark:hover:text-white ms-5">
                          <svg class="w-4 h-4" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 20 20">
                                <path fill-rule="evenodd" d="M10 .333A9.911 9.911 0 0 0 6.866 19.65c.5.092.678-.215.678-.477 0-.237-.01-1.017-.014-1.845-2.757.6-3.338-1.169-3.338-1.169a2.627 2.627 0 0 0-1.1-1.451c-.9-.615.07-.6.07-.6a2.084 2.084 0 0 1 1.518 1.021 2.11 2.11 0 0 0 2.884.823c.044-.503.268-.973.63-1.325-2.2-.25-4.516-1.1-4.516-4.9A3.832 3.832 0 0 1 4.7 7.068a3.56 3.56 0 0 1 .095-2.623s.832-.266 2.726 1.016a9.409 9.409 0 0 1 4.962 0c1.89-1.282 2.717-1.016 2.717-1.016.366.83.402 1.768.1 2.623a3.827 3.827 0 0 1 1.02 2.659c0 3.807-2.319 4.644-4.525 4.889a2.366 2.366 0 0 1 .673 1.834c0 1.326-.012 2.394-.012 2.72 0 .263.18.572.681.475A9.911 9.911 0 0 0 10 .333Z" clip-rule="evenodd"/>
                          </svg>
                          <span class="sr-only">GitHub account</span>
                      </a>
                  </div>
              </div>
          </div>
      </footer>
      </div>
    </section>
    
  );
};

export default Home;