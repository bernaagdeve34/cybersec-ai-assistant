const express = require('express');
const axios = require('axios');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// React frontend'den gelen POST /ask isteklerini FastAPI'ye ilet
app.post('/ask', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:8000/ask', req.body);
    res.json(response.data);
  } catch (err) {
    res.status(500).json({ error: 'Backend error', details: err.message });
  }
});

// Sağlık kontrolü için basit bir endpoint
app.get('/', (req, res) => {
  res.send('Node.js proxy backend çalışıyor!');
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Node.js backend listening on port ${PORT}`);
});

