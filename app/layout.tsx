import './globals.css'
import { Inter } from 'next/font/google'
import { ThemeProvider } from "@/components/theme-provider"
import { GamificationProvider } from '@/contexts/gamification-context'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'MCAT Study App',
  description: 'An app for college students studying for the MCAT',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
          <GamificationProvider>
            {children}
          </GamificationProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}

