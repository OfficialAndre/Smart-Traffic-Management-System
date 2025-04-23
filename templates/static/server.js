const express = require('express');
const path = require('path');
const app = express();
const port = 3000;

// Serve static files from the current directory
app.use(express.static(path.join(__dirname)));

// Set the views directory
app.set('views', path.join(__dirname, 'templates'));

// Set the view engine to serve HTML files
app.set('view engine', 'html');

// Serve the index.html file on the root URL
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

app.get('/traffic-data', (req, res) => {
    const data = { vehicleCount: Math.floor(Math.random() * 100) };
    res.json(data);
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
