"use client"

import React, { createContext, useContext, useReducer, useEffect } from 'react';
import { 
  XP_RULES, 
  LEVEL_THRESHOLDS, 
  calculateLevel, 
  xpForNextLevel,
  calculateStreakBonus,
  checkDailyGoals
} from '@/utils/gamification';
import { offlineStorage } from '../utils/offlineStorage';

type GamificationState = {
  xp: number;
  level: number;
  streakDays: number;
  lastLoginDate: string;
  achievements: string[];
  dailyStats: {
    flashcardsReviewed: number;
    practiceQuestions: number;
    studyMinutes: number;
    dailyXp: number;
  };
};

type GamificationAction =
  | { type: 'ADD_XP'; payload: number }
  | { type: 'UPDATE_STREAK' }
  | { type: 'UNLOCK_ACHIEVEMENT'; payload: string }
  | { type: 'UPDATE_DAILY_STATS'; payload: Partial<GamificationState['dailyStats']> }
  | { type: 'RESET_DAILY_STATS' };

const initialState: GamificationState = {
  xp: 0,
  level: 1,
  streakDays: 0,
  lastLoginDate: new Date().toISOString().split('T')[0],
  achievements: [],
  dailyStats: {
    flashcardsReviewed: 0,
    practiceQuestions: 0,
    studyMinutes: 0,
    dailyXp: 0,
  },
};

const GamificationContext = createContext<{
  state: GamificationState;
  addXP: (amount: number) => void;
  reviewFlashcard: () => void;
  completePracticeTest: (score: number) => void;
  updateStudyTime: (minutes: number) => void;
} | null>(null);

function gamificationReducer(state: GamificationState, action: GamificationAction): GamificationState {
  switch (action.type) {
    case 'ADD_XP': {
      const newXp = state.xp + action.payload;
      const streakBonus = calculateStreakBonus(state.streakDays);
      const bonusXp = Math.floor(action.payload * (streakBonus - 1));
      const totalNewXp = newXp + bonusXp;
      const newLevel = calculateLevel(totalNewXp);
      
      return {
        ...state,
        xp: totalNewXp,
        level: newLevel,
        dailyStats: {
          ...state.dailyStats,
          dailyXp: state.dailyStats.dailyXp + action.payload + bonusXp,
        },
      };
    }
    
    case 'UPDATE_STREAK': {
      const today = new Date().toISOString().split('T')[0];
      const lastLogin = new Date(state.lastLoginDate);
      const diffDays = Math.floor((new Date().getTime() - lastLogin.getTime()) / (1000 * 60 * 60 * 24));
      
      if (diffDays === 1) {
        // Consecutive day
        return {
          ...state,
          streakDays: state.streakDays + 1,
          lastLoginDate: today,
        };
      } else if (diffDays > 1) {
        // Streak broken
        return {
          ...state,
          streakDays: 1,
          lastLoginDate: today,
        };
      }
      return state;
    }
    
    case 'UNLOCK_ACHIEVEMENT':
      if (state.achievements.includes(action.payload)) {
        return state;
      }
      return {
        ...state,
        achievements: [...state.achievements, action.payload],
      };
      
    case 'UPDATE_DAILY_STATS':
      return {
        ...state,
        dailyStats: {
          ...state.dailyStats,
          ...action.payload,
        },
      };
      
    case 'RESET_DAILY_STATS':
      return {
        ...state,
        dailyStats: initialState.dailyStats,
      };
      
    default:
      return state;
  }
}

export function GamificationProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(gamificationReducer, initialState);
  
  // Load state from storage on mount
  useEffect(() => {
    async function loadState() {
      const savedState = await offlineStorage.getItem('gamification');
      if (savedState) {
        Object.entries(savedState).forEach(([key, value]) => {
          dispatch({ type: 'UPDATE_DAILY_STATS', payload: { [key]: value } });
        });
      }
    }
    loadState();
  }, []);
  
  // Save state to storage on change
  useEffect(() => {
    offlineStorage.setItem('gamification', state);
  }, [state]);
  
  // Check for streak updates on mount and date change
  useEffect(() => {
    dispatch({ type: 'UPDATE_STREAK' });
    
    // Check for date change every minute
    const interval = setInterval(() => {
      const today = new Date().toISOString().split('T')[0];
      if (today !== state.lastLoginDate) {
        dispatch({ type: 'UPDATE_STREAK' });
        dispatch({ type: 'RESET_DAILY_STATS' });
      }
    }, 60000);
    
    return () => clearInterval(interval);
  }, [state.lastLoginDate]);
  
  const addXP = (amount: number) => {
    dispatch({ type: 'ADD_XP', payload: amount });
  };
  
  const reviewFlashcard = () => {
    dispatch({ 
      type: 'UPDATE_DAILY_STATS', 
      payload: { flashcardsReviewed: state.dailyStats.flashcardsReviewed + 1 } 
    });
    addXP(XP_RULES.FLASHCARD_REVIEW);
  };
  
  const completePracticeTest = (score: number) => {
    dispatch({ 
      type: 'UPDATE_DAILY_STATS', 
      payload: { practiceQuestions: state.dailyStats.practiceQuestions + 1 } 
    });
    const xpEarned = Math.floor(XP_RULES.PRACTICE_TEST_COMPLETION * (score / 100));
    addXP(xpEarned);
  };
  
  const updateStudyTime = (minutes: number) => {
    dispatch({ 
      type: 'UPDATE_DAILY_STATS', 
      payload: { studyMinutes: state.dailyStats.studyMinutes + minutes } 
    });
  };
  
  return (
    <GamificationContext.Provider 
      value={{ 
        state, 
        addXP, 
        reviewFlashcard, 
        completePracticeTest, 
        updateStudyTime 
      }}
    >
      {children}
    </GamificationContext.Provider>
  );
}

export function useGamification() {
  const context = useContext(GamificationContext);
  if (!context) {
    throw new Error('useGamification must be used within a GamificationProvider');
  }
  return context;
}

