/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        aura: {
          50: '#f4f6fb',
          100: '#e8edf7',
          200: '#cbd7ee',
          300: '#9eb8e1',
          400: '#6b93d1',
          500: '#4774c2',
          600: '#345ba6',
          700: '#2b4987',
          800: '#273e6e',
          900: '#23365c',
          950: '#0f172a',
        },
      },
    },
  },
  plugins: [],
}
