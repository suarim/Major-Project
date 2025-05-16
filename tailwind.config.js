/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f7ff',
          100: '#e0effe',
          200: '#bae0fd',
          300: '#7dccfd',
          400: '#36b3f9',
          500: '#0c95e9',
          600: '#0077cc',
          700: '#0061a6',
          800: '#00528a',
          900: '#064673',
        },
        secondary: {
          50: '#effbff',
          100: '#daf6fe',
          200: '#bcedfd',
          300: '#8adffa',
          400: '#4cccf5',
          500: '#22b1e9',
          600: '#0d93cc',
          700: '#0a75a6',
          800: '#0e608a',
          900: '#115173',
        },
        accent: {
          50: '#f9f7ff',
          100: '#f4efff',
          200: '#eaddff',
          300: '#dabeff',
          400: '#c58fff',
          500: '#a85bff',
          600: '#9437ff',
          700: '#8521ef',
          800: '#6c19c3',
          900: '#59189f',
        },
        success: {
          50: '#effef7',
          100: '#dafeef',
          200: '#b8fadb',
          300: '#84f4bf',
          400: '#48e79a',
          500: '#1fcf76',
          600: '#13aa5d',
          700: '#13874d',
          800: '#146a40',
          900: '#135836',
        },
        warning: {
          50: '#fffbea',
          100: '#fff3c4',
          200: '#fce588',
          300: '#fadb5f',
          400: '#f7c948',
          500: '#f0b429',
          600: '#de911d',
          700: '#cb6e17',
          800: '#b44d12',
          900: '#8d2b0b',
        },
        error: {
          50: '#fef2f2',
          100: '#fde8e8',
          200: '#fbd5d5',
          300: '#f8b4b4',
          400: '#f98080',
          500: '#f05252',
          600: '#e02424',
          700: '#c81e1e',
          800: '#9b1c1c',
          900: '#771d1d',
        },
        neutral: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
      },
      fontFamily: {
        sans: [
          'Inter var', 
          'system-ui', 
          'sans-serif'
        ],
      },
      spacing: {
        '1': '8px',
        '2': '16px',
        '3': '24px',
        '4': '32px',
        '5': '40px',
        '6': '48px',
      },
      animation: {
        'typing': 'typing 3.5s steps(40, end)',
        'blink': 'blink .75s step-end infinite',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        typing: {
          'from': { width: '0' },
          'to': { width: '100%' }
        },
        blink: {
          'from, to': { borderColor: 'transparent' },
          '50%': { borderColor: 'currentColor' }
        }
      }
    },
  },
  plugins: [],
}