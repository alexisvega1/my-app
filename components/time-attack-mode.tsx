"use client"

import { useState, useEffect } from 'react';
import { Timer, Zap, Trophy, TrendingUp, Star, Medal, Crown, Activity, Check, X } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

// Question bank with various types
const questionBank = [
  {
    id: 1,
    type: 'main_idea',
    text: "What is the primary argument of the passage?",
    options: [
      "The evolution of scientific methodology",
      "The impact of technological advancement",
      "The relationship between theory and practice",
      "The role of empirical evidence"
    ],
    correct: 0,
    difficulty: 2,
    points: 100
  },
  {
    id: 2,
    type: 'inference',
    text: "Based on the passage, what can be inferred about the author's perspective?",
    options: [
      "Strongly supportive of traditional methods",
      "Critical of modern approaches",
      "Balanced between old and new",
      "Advocating for radical change"
    ],
    correct: 2,
    difficulty: 3,
    points: 150
  },
  {
    id: 3,
    type: 'tone',
    text: "The author's tone can best be described as:",
    options: [
      "Analytical and objective",
      "Critical and dismissive",
      "Enthusiastic and supportive",
      "Cautious and skeptical"
    ],
    correct: 0,
    difficulty: 2,
    points: 100
  }
  // Add more questions here...
];

// Achievement definitions
const achievements = {
  speed_demon: { name: "Speed Demon", xp: 500, condition: "Answer 5 questions in under 30 seconds each" },
  perfect_streak: { name: "Perfect Streak", xp: 1000, condition: "Get 10 correct answers in a row" },
  master_mind: { name: "Master Mind", xp: 2000, condition: "Score over 5000 points in one session" }
};

interface TimeAttackModeProps {
  onUpdateStats: (stats: { xp: number; correct: number; total: number }) => void;
  addXP: (amount: number) => void;
}

export default function TimeAttackMode({ onUpdateStats, addXP }: TimeAttackModeProps) {
  const [timeLeft, setTimeLeft] = useState(120);
  const [gameState, setGameState] = useState('idle');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [metrics, setMetrics] = useState({
    questionsAttempted: 0,
    questionsCorrect: 0,
    streak: 0,
    totalScore: 0,
    xp: 0,
    achievements: []
  });
  const [showFeedback, setShowFeedback] = useState(false);
  const [lastAnswerCorrect, setLastAnswerCorrect] = useState(false);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (gameState === 'active' && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft(t => t - 1);
      }, 1000);
    } else if (timeLeft === 0 && gameState === 'active') {
      endGame();
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [gameState, timeLeft]);

  // Get streak multiplier
  const getStreakMultiplier = () => {
    if (metrics.streak >= 10) return 3;
    if (metrics.streak >= 5) return 2;
    if (metrics.streak >= 3) return 1.5;
    return 1;
  };

  const handleAnswer = async (selectedIndex: number) => {
    const currentQuestion = questionBank[currentQuestionIndex];
    const isCorrect = selectedIndex === currentQuestion.correct;
    
    // Show immediate feedback
    setLastAnswerCorrect(isCorrect);
    setShowFeedback(true);

    // Calculate points with multipliers
    const basePoints = currentQuestion.points;
    const streakMultiplier = getStreakMultiplier();
    const difficultyMultiplier = currentQuestion.difficulty * 0.5;
    const totalPoints = Math.round(basePoints * streakMultiplier * difficultyMultiplier);

    // Update metrics
    setMetrics(prev => ({
      ...prev,
      questionsAttempted: prev.questionsAttempted + 1,
      questionsCorrect: prev.questionsCorrect + (isCorrect ? 1 : 0),
      streak: isCorrect ? prev.streak + 1 : 0,
      totalScore: prev.totalScore + (isCorrect ? totalPoints : 0),
      xp: prev.xp + (isCorrect ? Math.round(totalPoints * 0.1) : 0)
    }));

    // Show feedback briefly
    await new Promise(resolve => setTimeout(resolve, 500));
    setShowFeedback(false);

    // Move to next question
    if (currentQuestionIndex < questionBank.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    } else {
      setCurrentQuestionIndex(0); // Loop back to start if needed
    }

    // Update stats
    onUpdateStats({
      xp: isCorrect ? Math.round(totalPoints * 0.1) : 0,
      correct: isCorrect ? 1 : 0,
      total: 1
    });

    // Add XP
    if (isCorrect) {
      addXP(Math.round(totalPoints * 0.1));
    }
  };

  const renderQuestion = () => {
    const question = questionBank[currentQuestionIndex];
    return (
      <Card className={`bg-gray-800 border-0 transition-all duration-300 ${
        showFeedback ? (lastAnswerCorrect ? 'bg-green-800/20' : 'bg-red-800/20') : ''
      }`}>
        <CardContent className="p-6">
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-400">Question Type: {question.type}</span>
              <span className="text-sm text-gray-400">
                Points: {question.points} x{getStreakMultiplier()}
              </span>
            </div>
            <h3 className="text-lg font-medium">{question.text}</h3>
          </div>
          <div className="grid grid-cols-1 gap-3">
            {question.options.map((option, index) => (
              <Button
                key={index}
                onClick={() => handleAnswer(index)}
                variant="outline"
                className={`
                  w-full p-4 text-left justify-start h-auto
                  ${showFeedback && index === question.correct ? 'bg-green-500/20' : ''} 
                `}
              >
                {option}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  };

  const startGame = () => {
    setTimeLeft(120);
    setGameState('active');
    setCurrentQuestionIndex(0);
    setMetrics({
      questionsAttempted: 0,
      questionsCorrect: 0,
      streak: 0,
      totalScore: 0,
      xp: 0,
      achievements: []
    });
  };

  const endGame = () => {
    setGameState('results');
    // You can add more end-game logic here, such as saving high scores or updating overall stats
  };

  return (
    <div className="space-y-6">
      {/* Header with XP and Streak */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-4">
          <Timer className={timeLeft < 30 ? 'text-red-500 animate-pulse' : 'text-blue-500'} />
          <span className="text-2xl font-bold">
            {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, '0')}
          </span>
        </div>
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <Zap className="text-yellow-500" />
            <span className="text-xl font-bold">{metrics.xp} XP</span>
          </div>
          <div className="flex items-center space-x-2">
            <Trophy className="text-blue-500" />
            <span className="text-xl font-bold">
              {metrics.streak > 0 && `${metrics.streak}x (${getStreakMultiplier()})`}
            </span>
          </div>
        </div>
      </div>

      {/* Game Area */}
      {gameState === 'idle' && (
        <Button onClick={startGame} className="w-full py-8 text-xl">
          Start Time Attack
        </Button>
      )}
      {gameState === 'active' && renderQuestion()}
      {gameState === 'results' && (
        <Card className="bg-gray-800 border-0">
          <CardHeader>
            <CardTitle>Time's Up!</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl mb-4">Final Score: {metrics.totalScore}</div>
            <div className="text-xl mb-4">XP Earned: {metrics.xp}</div>
            <div className="text-lg mb-4">Questions Answered: {metrics.questionsAttempted}</div>
            <div className="text-lg mb-4">Correct Answers: {metrics.questionsCorrect}</div>
            <Button onClick={startGame} className="w-full">
              Play Again
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Feedback Overlay */}
      {showFeedback && (
        <div className="fixed inset-0 flex items-center justify-center pointer-events-none">
          <div className={`text-6xl transform scale-150 transition-transform ${
            lastAnswerCorrect ? 'text-green-500' : 'text-red-500'
          }`}>
            {lastAnswerCorrect ? <Check /> : <X />}
          </div>
        </div>
      )}
    </div>
  );
}

