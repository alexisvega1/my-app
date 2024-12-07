export const XP_RULES = {
  FLASHCARD_REVIEW: 10,
  PRACTICE_TEST_COMPLETION: 100,
  CORRECT_ANSWER: 20,
  DAILY_LOGIN: 50,
  STREAK_BONUS: 25,
  ACHIEVEMENT_UNLOCK: 200,
} as const;

// Level thresholds (XP needed for each level)
export const LEVEL_THRESHOLDS = Array.from({ length: 50 }, (_, i) => {
  // More gradual XP curve: each level requires slightly more XP than the last
  return Math.floor(100 * Math.pow(1.1, i));
});

// Streak rules
export const STREAK_RULES = {
  // Hours before streak resets (resets at midnight)
  RESET_HOURS: 24,
  // Bonus multipliers for streak milestones
  MULTIPLIERS: {
    7: 1.5,   // 1.5x bonus for 7-day streak
    30: 2.0,  // 2x bonus for 30-day streak
    100: 3.0, // 3x bonus for 100-day streak
  },
  // Minimum XP required to maintain streak
  MIN_DAILY_XP: 50,
} as const;

// Achievement definitions
export const ACHIEVEMENTS = {
  BEGINNER: {
    id: 'beginner',
    title: 'First Steps',
    description: 'Complete your first practice test',
    xpReward: 100,
    icon: 'Baby',
  },
  STREAK_WARRIOR: {
    id: 'streak_warrior',
    title: 'Streak Warrior',
    description: 'Maintain a 7-day study streak',
    xpReward: 200,
    icon: 'Flame',
  },
  FLASH_MASTER: {
    id: 'flash_master',
    title: 'Flash Master',
    description: 'Review 100 flashcards in one day',
    xpReward: 300,
    icon: 'Zap',
  },
  PERFECT_SCORE: {
    id: 'perfect_score',
    title: 'Perfect Score',
    description: 'Get 100% on a practice test',
    xpReward: 500,
    icon: 'Trophy',
  },
} as const;

// Daily goals
export const DAILY_GOALS = {
  FLASHCARDS_REVIEWED: 20,
  PRACTICE_QUESTIONS: 10,
  STUDY_MINUTES: 30,
  MIN_XP: 100,
} as const;

// Calculate level from XP
export function calculateLevel(xp: number): number {
  return LEVEL_THRESHOLDS.findIndex(threshold => xp < threshold) + 1;
}

// Calculate XP needed for next level
export function xpForNextLevel(currentXp: number): number {
  const currentLevel = calculateLevel(currentXp);
  return LEVEL_THRESHOLDS[currentLevel] - currentXp;
}

// Calculate streak bonus
export function calculateStreakBonus(streakDays: number): number {
  const multiplier = Object.entries(STREAK_RULES.MULTIPLIERS)
    .reverse()
    .find(([days]) => streakDays >= Number(days))?.[1] || 1;
  
  return multiplier;
}

// Check if daily goals are met
export function checkDailyGoals(stats: {
  flashcardsReviewed: number;
  practiceQuestions: number;
  studyMinutes: number;
  dailyXp: number;
}): number {
  let progress = 0;
  
  if (stats.flashcardsReviewed >= DAILY_GOALS.FLASHCARDS_REVIEWED) progress += 25;
  if (stats.practiceQuestions >= DAILY_GOALS.PRACTICE_QUESTIONS) progress += 25;
  if (stats.studyMinutes >= DAILY_GOALS.STUDY_MINUTES) progress += 25;
  if (stats.dailyXp >= DAILY_GOALS.MIN_XP) progress += 25;
  
  return progress;
}

