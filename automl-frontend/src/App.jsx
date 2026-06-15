import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/layout/Layout";
import Dashboard from "./pages/Dashboard";
import DatasetUpload from "./pages/DatasetUpload";
import Models from "./pages/Models";
import Training from "./pages/Training";
import Inference from "./pages/Inference";
import Monitoring from "./pages/Monitoring";
import Automation from "./pages/Automation";
import Policies from "./pages/Policies";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<DatasetUpload />} />
          <Route path="/models" element={<Models />} />
          <Route path="/training" element={<Training />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/automation" element={<Automation />} />
          <Route path="/policies" element={<Policies />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

