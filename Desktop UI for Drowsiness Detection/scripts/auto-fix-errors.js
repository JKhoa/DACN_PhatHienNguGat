/**
 * Auto-fix script for common errors in the application
 * This script automatically fixes common React, TypeScript, and import errors
 */

const fs = require('fs');
const path = require('path');

const errorsFixed = [];

// Helper function to read file
function readFile(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    console.error(`Error reading ${filePath}:`, error.message);
    return null;
  }
}

// Helper function to write file
function writeFile(filePath, content) {
  try {
    fs.writeFileSync(filePath, content, 'utf8');
    errorsFixed.push(`Fixed: ${filePath}`);
    return true;
  } catch (error) {
    console.error(`Error writing ${filePath}:`, error.message);
    return false;
  }
}

// Fix: Add React import if missing and JSX is used
function fixReactImport(filePath, content) {
  const hasJSX = /<[A-Z]/.test(content);
  const hasReactImport = /import\s+React|import\s+\*\s+as\s+React|from\s+['"]react['"]/.test(content);
  
  if (hasJSX && !hasReactImport && !content.includes('import React')) {
    // Check if there's an import statement
    if (content.includes("import")) {
      // Find first import and add React import before it
      content = "import React from 'react';\n" + content;
      writeFile(filePath, content);
      console.log(`✅ Added React import to ${path.basename(filePath)}`);
      return true;
    }
  }
  return false;
}

// Fix: Ensure useRef is imported
function fixUseRefImport(filePath, content) {
  const usesUseRef = /useRef\(/g.test(content);
  const hasUseRefImport = /import.*useRef.*from/.test(content);
  
  if (usesUseRef && !hasUseRefImport) {
    // Try to add useRef to existing React import
    if (content.includes("import") && content.includes("from 'react'")) {
      content = content.replace(
        /import\s+(\{[^}]*\})\s+from\s+['"]react['"]/,
        (match, imports) => {
          if (!imports.includes('useRef')) {
            return `import ${imports.replace('}', ', useRef }')} from 'react'`;
          }
          return match;
        }
      );
      
      if (!content.includes('useRef')) {
        // Add separate import if needed
        content = content.replace(
          /import\s+React.*from\s+['"]react['"]/,
          (match) => match + "\nimport { useRef } from 'react';"
        );
      }
      
      writeFile(filePath, content);
      console.log(`✅ Added useRef import to ${path.basename(filePath)}`);
      return true;
    }
  }
  return false;
}

// Fix: Remove type arguments from hooks (if causing issues)
function fixHookTypeArguments(filePath, content) {
  // Only fix if there are actual errors - this is a conservative approach
  // Type arguments are actually valid in modern TypeScript
  return false;
}

// Fix: Add labels to form elements
function fixFormLabels(filePath, content) {
  let fixed = false;
  
  // Fix input elements without labels
  content = content.replace(
    /<input\s+([^>]*)\s*\/>/g,
    (match, attrs) => {
      if (!attrs.includes('aria-label') && 
          !attrs.includes('aria-labelledby') && 
          !attrs.includes('title') &&
          !attrs.includes('placeholder')) {
        // Add aria-label
        fixed = true;
        return `<input ${attrs} aria-label="Input field" />`;
      }
      return match;
    }
  );
  
  if (fixed) {
    writeFile(filePath, content);
    console.log(`✅ Added labels to form elements in ${path.basename(filePath)}`);
    return true;
  }
  return false;
}

// Main function to scan and fix files
function scanAndFix(directory) {
  const files = fs.readdirSync(directory, { recursive: true });
  
  for (const file of files) {
    const filePath = path.join(directory, file);
    
    // Only process TypeScript/JavaScript files
    if (!filePath.match(/\.(tsx?|jsx?)$/)) continue;
    
    // Skip node_modules
    if (filePath.includes('node_modules')) continue;
    
    let content = readFile(filePath);
    if (!content) continue;
    
    let modified = false;
    
    // Apply all fixes
    if (fixReactImport(filePath, content)) {
      content = readFile(filePath); // Re-read after fix
      modified = true;
    }
    
    if (fixUseRefImport(filePath, content)) {
      content = readFile(filePath);
      modified = true;
    }
    
    if (fixFormLabels(filePath, content)) {
      modified = true;
    }
  }
  
  console.log(`\n✅ Fixed ${errorsFixed.length} issues`);
  if (errorsFixed.length > 0) {
    console.log('\nFixed files:');
    errorsFixed.forEach(f => console.log(`  - ${f}`));
  }
}

// Run the fix script
const srcDir = path.join(__dirname, '../src');
console.log('🔍 Scanning for errors...\n');
scanAndFix(srcDir);

