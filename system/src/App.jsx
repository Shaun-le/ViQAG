import {Route, BrowserRouter as Router, Routes} from 'react-router-dom';
import './index.css'
import Navbar from './component/Navbar'
import {Home, Pricing, About, Blog, Contact, Login, Sign_up} from './pages'

const App = () => {
  return (
    <main className='bg-slate-300/20'>
        <Router>
            <Navbar/>
            <Routes>
                <Route path="/" element ={<Home />} />
                <Route path="/pricing" element ={<Pricing />} />
                <Route path="/about" element ={<About />} />
                <Route path="/blog" element ={<Blog />} />
                <Route path="/contact" element ={<Contact />} />
                <Route path="/login" element ={<Login />} />
                <Route path="/signup" element ={<Sign_up />} />
            </Routes>
        </Router>
    </main>
  )
}

export default App