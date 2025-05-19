# ASTER Web Explorer - Frontend

The frontend component of the ASTER Web Explorer application, providing an interactive web interface for processing, analyzing, and visualizing ASTER satellite data for geological and mineral exploration purposes.

## Features

- Interactive map visualization using Leaflet
- Upload and process raw ASTER data
- Visualize mineral and alteration maps
- Perform gold prospectivity analysis
- Generate and view comprehensive reports
- User-friendly interface for ASTER data exploration

## Technology Stack

- **React**: UI framework
- **Vite**: Build tool
- **React Router**: Navigation
- **Leaflet/React-Leaflet**: Interactive mapping
- **TailwindCSS**: Styling
- **Shadcn/UI**: UI component library
- **Recharts**: Data visualization
- **Lucide**: Icon library

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Installation

1. Clone the repository
2. Navigate to the frontend directory:
   ```bash
   cd aster-web-app/frontend
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Create a `.env` file with the following content:
   ```
   VITE_API_BASE_URL=http://localhost:5000/api
   ```

### Development

Run the development server:

```bash
npm run dev
```

The application will be available at http://localhost:3000

### Building for Production

```bash
npm run build
```

This will create an optimized build in the `dist` directory.

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── ui/          # Shadcn UI components
│   │   └── ...          # Application-specific components
│   ├── hooks/           # Custom React hooks
│   ├── pages/           # Page components
│   ├── services/        # API service functions
│   ├── utils/           # Utility functions
│   ├── contexts/        # React contexts
│   ├── enums/           # Enumeration definitions
│   ├── styles/          # CSS and style files
│   ├── App.jsx          # Main App component
│   └── index.jsx        # Entry point
├── .env                 # Environment variables
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
└── vite.config.js       # Vite configuration
```

## Integration with Backend

The frontend communicates with the backend API through the services defined in `src/services/api.js`. All API requests are routed through this service layer, which provides a clean interface for interacting with the backend.

## Component Structure

- **Layout Components**: Define the overall structure of the application
- **Page Components**: Specific views for different features
- **UI Components**: Reusable building blocks
- **Map Components**: Map visualization and controls
- **Form Components**: Input and processing controls

## Styling

The application uses TailwindCSS for styling, with custom components from the shadcn/ui library. These components are styled using a combination of Tailwind classes and customized to match the application's design system.