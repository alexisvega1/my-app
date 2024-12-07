import { BarChart2, Brain, Settings, LogOut } from 'lucide-react'

interface SidebarProps {
  activeSection: string
  setActiveSection: (section: string) => void
}

export function Sidebar({ activeSection, setActiveSection }: SidebarProps) {
  return (
    <div className="w-20 bg-card border-r border-border flex flex-col items-center py-6">
      <div className="flex-1 space-y-8">
        <button 
          onClick={() => setActiveSection('dashboard')}
          className={`p-3 rounded-xl transition-colors ${
            activeSection === 'dashboard' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
          }`}
          aria-label="Dashboard"
        >
          <BarChart2 size={24} />
        </button>
        <button 
          onClick={() => setActiveSection('flashcards')}
          className={`p-3 rounded-xl transition-colors ${
            activeSection === 'flashcards' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
          }`}
          aria-label="Flashcards"
        >
          <Brain size={24} />
        </button>
        <button 
          onClick={() => setActiveSection('analytics')}
          className={`p-3 rounded-xl transition-colors ${
            activeSection === 'analytics' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
          }`}
          aria-label="Analytics"
        >
          <BarChart2 size={24} />
        </button>
      </div>
      <div className="mt-auto space-y-6">
        <button 
          onClick={() => setActiveSection('settings')}
          className={`p-3 rounded-xl transition-colors ${
            activeSection === 'settings' ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground hover:bg-accent'
          }`}
          aria-label="Settings"
        >
          <Settings size={24} />
        </button>
        <button 
          className="p-3 rounded-xl text-muted-foreground hover:text-foreground hover:bg-accent"
          aria-label="Log out"
        >
          <LogOut size={24} />
        </button>
      </div>
    </div>
  )
}

