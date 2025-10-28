#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// Function to fix import statements
function fixImports(content) {
  // Remove version numbers from import statements
  return content.replace(/from\s+["']([^"']+)@[0-9.]+["']/g, 'from "$1"');
}

// Function to recursively process files
function processDirectory(dirPath) {
  const files = fs.readdirSync(dirPath);
  
  files.forEach(file => {
    const filePath = path.join(dirPath, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      processDirectory(filePath);
    } else if (file.endsWith('.tsx') || file.endsWith('.ts')) {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const fixedContent = fixImports(content);
        
        if (content !== fixedContent) {
          fs.writeFileSync(filePath, fixedContent, 'utf8');
          console.log(`Fixed imports in: ${filePath}`);
        }
      } catch (error) {
        console.error(`Error processing ${filePath}:`, error.message);
      }
    }
  });
}

// Process the src directory
const srcPath = path.join(__dirname, 'src');
if (fs.existsSync(srcPath)) {
  console.log('Fixing import statements...');
  processDirectory(srcPath);
  console.log('Done!');
} else {
  console.error('src directory not found');
}

