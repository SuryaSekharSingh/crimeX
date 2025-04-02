import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = createServer(app);
const io = new Server(server);

app.use(express.static('public'));

let authoritiesData = {};
async function loadAuthorities() {
  try {
    const data = await fs.readFile(path.join(__dirname, 'authorities.json'), 'utf8');
    authoritiesData = JSON.parse(data);
    console.log('Loaded authorities data');
  } catch (error) {
    console.error('Error loading authorities data:', error);
  }
}
await loadAuthorities();

// Notification history storage
const notificationHistory = new Map();

// Enhanced department matching
function determineDepartment(report) {
  const content = `${report.type} ${report.description}`.toLowerCase();
  
  // Check for exact matches first
  for (const [dept, data] of Object.entries(authoritiesData)) {
    if (data.keywords.some(keyword => 
      new RegExp(`\\b${keyword}\\b`, 'i').test(content)
    )) {
      return dept;
    }
  }
  
  // Fallback to keyword matching
  for (const [dept, data] of Object.entries(authoritiesData)) {
    if (data.keywords.some(keyword => content.includes(keyword))) {
      return dept;
    }
  }
  
  return 'General';
}

io.on('connection', (socket) => {
  console.log('New client connected');

  socket.on('registerAuthority', (department) => {
    socket.join(department);
    socket.department = department;
    console.log(`Authority registered for ${department}`);
    
    // Send pending notifications
    if (notificationHistory.has(department)) {
      notificationHistory.get(department).forEach(notification => {
        socket.emit('newAlert', notification);
      });
    }
  });

  socket.on('crimeReport', (report) => {
    const department = determineDepartment(report);
    console.log(`Routing to ${department}`);
    
    const notification = {
      ...report,
      department,
      id: Date.now().toString(36) + Math.random().toString(36).substr(2)
    };

    // Store notification
    if (!notificationHistory.has(department)) {
      notificationHistory.set(department, []);
    }
    notificationHistory.get(department).push(notification);
    
    // Notify department and broadcast
    io.to(department).emit('newAlert', notification);
    socket.emit('reportAck', `Report forwarded to ${authoritiesData[department].name}`);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});