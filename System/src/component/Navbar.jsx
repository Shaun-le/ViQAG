import { useState } from 'react';
import { NavLink } from 'react-router-dom';

const Navbar = () => {

  return (
    <header>
      <nav className='fixed top-0 left-0 right-0 z-50 flex text-lg gap-7 font-medium items-center justify-between bg-white py-3'>
        <div className='relative'>
          <NavLink to='/' className='rounded-lg items-center justify-center flex font-bold px-8'>
            <p>
              <img src="./src/Figs/Logo/Logo_NLU-01.png" alt="" width="60" height="60"></img>
            </p>
          </NavLink>
        </div>
        {/*<div className='flex items-center gap-7' style={{ fontFamily: 'IBM Plex Mono' }}>
          <NavLink to="/pricing">
            <a className='text-black hover:bg-gray-700 hover:text-white rounded-md px-3 py-2 text-lg font-medium'>
              Pricing
            </a>
          </NavLink>
          <NavLink to="/about">
            <a className='text-black hover:bg-gray-700 hover:text-white rounded-md px-3 py-2 text-lg font-medium'>
              About
            </a>
          </NavLink>
          <NavLink to="/blog">
            <a className='text-black hover:bg-gray-700 hover:text-white rounded-md px-3 py-2 text-lg font-medium'>
              Blog
            </a>
          </NavLink>
          <NavLink to="/contact">
            <a className='text-black hover:bg-gray-700 hover:text-white rounded-md px-3 py-2 text-lg font-medium'>
              Contact
            </a>
          </NavLink>
        </div>*/}
        {/*<div className='hidden lg:flex items-center gap-4 px-8' style={{ fontFamily: 'IBM Plex Mono' }}>
          <NavLink to="/login">
            <a className='text-black hover:bg-gray-700 hover:text-white rounded-md px-3 py-2 text-lg font-medium '>
              Login
            </a>
          </NavLink>
          <NavLink to="/signup">
            <a className='bg-gray-900 text-white rounded-md px-3 py-2 text-lg font-medium'>
              Sign Up
            </a>
          </NavLink>
      </div>*/}
      </nav>
    </header>
  );
};

export default Navbar;