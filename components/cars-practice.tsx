"use client"

import { useState, useEffect } from 'react';
import { Home, Book, Clock, Highlighter, Plus, ChevronLeft, ChevronRight } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card } from "@/components/ui/card"

interface CARSPassage {
  id: number;
  title: string;
  source: string;
  content: string;
  keyTerms: string[];
  mainThemes: string[];
  difficulty: 'easy' | 'medium' | 'hard';
  timeEstimate: number; // in minutes
  category: 'humanities' | 'social_sciences' | 'natural_sciences';
  questionTypes: 'main_idea' | 'inference' | 'tone' | 'detail' | 'logic' | 'purpose';  // Updated questionTypes
  questions: {
    id: number;
    text: string;
    options: string[];
    correct: number;
    explanation: string;
    skillType: string; // Add skillType to questions
    difficulty: 'easy' | 'medium' | 'hard';
  }[];
}

interface StudyMetrics {
  timeSpent: number;
  correctAnswers: number;
  totalQuestions: number;
  skillPerformance: Record<string, number>;
  difficultyPerformance: Record<string, number>;
}

// Extended Helper Functions
const helpers = {
  calculateReadingTime: (content: string): number => {
    const wordsPerMinute = 250;
    const wordCount = content.split(' ').length;
    return Math.ceil(wordCount / wordsPerMinute);
  },

  getSkillLevelFromPerformance: (performance: number): string => {
    if (performance >= 85) return 'mastered';
    if (performance >= 70) return 'proficient';
    if (performance >= 50) return 'developing';
    return 'needs work';
  },

  generateProgressReport: (metrics: StudyMetrics): string => {
    const accuracy = (metrics.correctAnswers / metrics.totalQuestions) * 100;
    const timePerQuestion = metrics.timeSpent / metrics.totalQuestions;
    
    return `
      Overall Accuracy: ${accuracy.toFixed(1)}%
      Average Time per Question: ${timePerQuestion.toFixed(1)} seconds
      Skills Performance: ${Object.entries(metrics.skillPerformance)
        .map(([skill, score]) => `${skill}: ${score}%`)
        .join(', ')}
    `;
  },

  getRecommendedPassages: (metrics: StudyMetrics): CARSPassage[] => {
    // Logic to recommend passages based on performance
    return practiceContent.full.passages.filter(passage => 
      passage.difficulty === (metrics.accuracy < 70 ? 'easy' : 'medium')
    );
  }
};

// Expanded Full Section Content
const practiceContent = {
  full: {
    passages: [
      {
        id: 1,
        title: "The Ethics of Scientific Discovery",
        source: "Journal of Scientific Ethics",
        category: 'natural_sciences',
        difficulty: 'hard',
        timeEstimate: 10,
        content: `The process of scientific discovery has traditionally been viewed as an objective pursuit of truth, unencumbered by moral considerations. However, recent developments in fields such as genetic engineering, artificial intelligence, and neuroscience have challenged this perspective, raising fundamental questions about the relationship between scientific progress and ethical responsibility. The rapid pace of technological advancement has outstripped our ability to fully comprehend its implications, creating a gap between what we can do and what we should do.

        Consider the case of CRISPR gene editing technology. While it offers unprecedented potential for treating genetic diseases, it also raises profound ethical concerns about designer babies and genetic enhancement. Similarly, advances in artificial intelligence present opportunities for solving complex problems while simultaneously raising questions about privacy, autonomy, and the nature of consciousness itself.

        Scientists can no longer claim immunity from the ethical implications of their work. The traditional model of conducting research first and considering consequences later has become increasingly untenable. This shift requires a fundamental reconsideration of how we approach scientific research, suggesting the need for an integrated framework that considers ethical implications at every stage of the discovery process.`,
        keyTerms: ["scientific ethics", "genetic engineering", "artificial intelligence", "ethical responsibility"],
        mainThemes: ["ethics in science", "technological progress", "moral responsibility"],
        questionTypes: 'purpose', // Example question type
        questions: [
          {
            id: 1,
            text: "The author's primary purpose in writing this passage is to:",
            options: [
              "Criticize modern scientific practices",
              "Argue for the integration of ethics in scientific research",
              "Explain recent developments in genetic engineering",
              "Defend traditional scientific methods"
            ],
            correct: 1,
            explanation: "The author emphasizes the need to integrate ethical considerations into scientific research, particularly evident in the final paragraph.",
            skillType: "purpose", // Example skill type
            difficulty: "medium"
          },
          {
            id: 2,
            text: "Based on the passage, which of the following best represents the author's view of scientific progress?",
            options: [
              "It should be stopped until ethical frameworks are developed",
              "It is inherently unethical",
              "It requires simultaneous ethical consideration",
              "It should proceed without ethical constraints"
            ],
            correct: 2,
            explanation: "The author suggests that scientific progress and ethical considerations should occur simultaneously, rejecting the traditional model of 'conducting research first and considering consequences later.'",
            skillType: "inference", // Example skill type
            difficulty: "hard"
          }
        ]
      },
      {
        id: 2,
        title: "The Evolution of Cultural Memory",
        source: "Anthropological Quarterly",
        category: 'humanities',
        difficulty: 'medium',
        timeEstimate: 8,
        content: `Cultural memory, the way societies remember and interpret their collective past, has undergone significant transformation in the digital age. Traditional methods of preserving cultural heritage through oral histories, written documents, and physical artifacts are being supplemented and sometimes replaced by digital archives, social media, and virtual experiences. This shift raises questions about the authenticity, accessibility, and permanence of cultural memory.

        Digital preservation offers unprecedented opportunities for documenting and sharing cultural heritage. A single online repository can house thousands of historical documents, photographs, and recordings, making them accessible to anyone with an internet connection. However, this democratization of cultural memory comes with its own challenges. Digital formats can become obsolete, data can be lost or corrupted, and the sheer volume of information can make it difficult to distinguish significant cultural artifacts from ephemera.

        Moreover, the digital preservation of cultural memory introduces new questions about ownership and interpretation. Who has the right to digitize and share cultural artifacts? How do we ensure that digital representations accurately reflect the cultural context and significance of the original items? These questions become particularly pertinent when dealing with indigenous cultures and marginalized communities whose histories have often been documented and interpreted by outsiders.`,
        keyTerms: ["cultural memory", "digital preservation", "cultural heritage", "democratization"],
        mainThemes: ["digital transformation", "cultural preservation", "accessibility vs. authenticity"],
        questionTypes: 'main_idea', // Example question type
        questions: [
          {
            id: 1,
            text: "The passage suggests that digital preservation of cultural memory primarily differs from traditional methods in its:",
            options: [
              "Superior accuracy",
              "Greater accessibility and scope",
              "Better preservation quality",
              "Higher cultural authenticity"
            ],
            correct: 1,
            explanation: "The passage emphasizes how digital preservation allows for widespread access and can house vast amounts of information, contrasting with traditional methods.",
            skillType: "main_idea", // Example skill type
            difficulty: "medium"
          }
        ]
      },
      {
        id: 3,
        title: "The Philosophy of Time",
        source: "Metaphysical Review",
        category: 'humanities',
        difficulty: 'hard',
        timeEstimate: 9,
        content: `The nature of time has puzzled philosophers and scientists throughout history. While we experience time as a linear progression from past to future, this intuitive understanding has been challenged by both philosophical inquiry and modern physics. Some philosophers argue that our perception of time's flow is an illusion, suggesting that past, present, and future exist simultaneously in what is known as an "eternal present."

        This view, often called eternalism, contrasts sharply with presentism, which holds that only the present moment is real, with the past and future being merely conceptual constructs. The debate between these perspectives has profound implications for our understanding of causality, free will, and the nature of reality itself.

        Modern physics, particularly Einstein's theory of relativity, seems to support the eternalist view by showing that time is relative and that the distinction between past, present, and future depends on the observer's frame of reference. However, this scientific perspective creates new philosophical puzzles about consciousness and human experience of time.`,
        keyTerms: ["eternalism", "presentism", "relativity", "temporal perception"],
        mainThemes: ["nature of time", "philosophical perspectives", "scientific understanding"],
        questionTypes: 'analysis', // Example question type
        questions: [
          {
            id: 1,
            text: "The main contrast presented in the passage is between:",
            options: [
              "Science and philosophy",
              "Past and future",
              "Eternalism and presentism",
              "Reality and illusion"
            ],
            correct: 2,
            explanation: "The passage primarily contrasts eternalism (all times exist simultaneously) with presentism (only the present is real).",
            skillType: "main_idea", // Example skill type
            difficulty: "medium"
          },
          {
            id: 2,
            text: "According to the passage, Einstein's theory of relativity:",
            options: [
              "Definitively proves eternalism",
              "Supports presentism",
              "Provides evidence that appears to align with eternalism",
              "Resolves all philosophical questions about time"
            ],
            correct: 2,
            explanation: "The passage states that relativity 'seems to support' eternalism, suggesting alignment but not definitive proof.",
            skillType: "detail", // Example skill type
            difficulty: "hard"
          }
        ]
      },
      {
        id: 4,
        title: "The Economics of Attention",
        source: "Digital Culture Quarterly",
        category: 'social_sciences',
        difficulty: 'medium',
        timeEstimate: 8,
        content: `In the digital age, attention has become a valuable commodity, leading to what economists call the "attention economy." As information becomes increasingly abundant, human attention remains a finite resource, creating a new type of scarcity that businesses and content creators compete to capture.

        This competition has led to sophisticated algorithms designed to maximize user engagement, often at the expense of deeper, more meaningful interactions. The result is a paradox: while we have access to more information than ever before, our ability to process and benefit from this information may be diminishing.

        The implications of the attention economy extend beyond individual user experience to affect social structures, democratic processes, and mental health. As attention becomes increasingly monetized, questions arise about the ethical responsibilities of platforms and content creators in managing this valuable resource.`,
        keyTerms: ["attention economy", "user engagement", "information abundance"],
        mainThemes: ["digital economics", "social impact", "technological ethics"],
        questionTypes: 'analysis', // Example question type
        questions: [
          {
            id: 1,
            text: "The passage suggests that the primary challenge in the digital age is:",
            options: [
              "Lack of information",
              "Managing abundant information with limited attention",
              "Creating engaging content",
              "Developing better algorithms"
            ],
            correct: 1,
            explanation: "The passage emphasizes the scarcity of attention in contrast to abundant information.",
            skillType: "inference", // Example skill type
            difficulty: "medium"
          }
        ]
      }
    ]
  },
  targeted: {
    passages: [
      // ... Previous targeted practice content
    ]
  },
  quick: {
    passages: [
      // ... Previous quick practice content
    ]
  }
};


// Additional Helper Functions
export const getPassagesByCategory = (category: string): CARSPassage[] => {
  const allPassages = [...practiceContent.full.passages]; // Include passages from all modes if needed
  return allPassages.filter(passage => passage.category === category);
};

export const getProgressBySkill = (
  answers: Record<number, number>,
  questions: Array<any>
): Record<string, number> => {
  const skillStats: Record<string, {correct: number, total: number}> = {};
  
  questions.forEach(q => {
    if (!skillStats[q.skillType]) {
      skillStats[q.skillType] = {correct: 0, total: 0};
    }
    skillStats[q.skillType].total++;
    if (answers[q.id] === q.correct) {
      skillStats[q.skillType].correct++;
    }
  });

  return Object.entries(skillStats).reduce((acc, [skill, stats]) => {
    acc[skill] = (stats.correct / stats.total) * 100;
    return acc;
  }, {} as Record<string, number>);
};

export const getRecommendedPractice = (
  metrics: StudyMetrics
): {passages: CARSPassage[], focus: string[]} => {
  const weakestSkills = Object.entries(metrics.skillPerformance)
    .sort(([,a], [,b]) => a - b)
    .slice(0, 2)
    .map(([skill]) => skill);

  return {
    passages: practiceContent.full.passages.filter(p => 
      p.questions.some(q => weakestSkills.includes(q.skillType))
    ),
    focus: weakestSkills
  };
};

const XP_RULES = {
  CORRECT_ANSWER: 10,
  // Add more XP rules as needed
};

interface CARSPracticeProps {
  onExitToMenu: () => void;
  onUpdateStats: (stats: { xp: number; correct: number; total: number }) => void;
  addXP: (amount: number) => void;
}

export default function CARSPractice({ onExitToMenu, onUpdateStats, addXP }: CARSPracticeProps) {
  const [mode, setMode] = useState(null);
  const [passages, setPassages] = useState<CARSPassage[]>([practiceContent.full.passages[0]]); // Updated initial state
  const [currentPassageIndex, setCurrentPassageIndex] = useState(0);
  const [currentPassage, setCurrentPassage] = useState<CARSPassage | null>(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [practiceStarted, setPracticeStarted] = useState(false);
  const [answeredQuestions, setAnsweredQuestions] = useState<{[key: number]: number}>({});
  const [hasSubmitted, setHasSubmitted] = useState(false);
  const [progress, setProgress] = useState({ // Updated progress state initialization
    accuracy: 0,
    streak: 0,
    masteryPoints: 0,
    xp: 0,
    level: 0,
    totalAnswered: 0,
    totalCorrect: 0
  });
  const [startTime, setStartTime] = useState(0); // Added startTime state
  const [studyMetrics, setStudyMetrics] = useState<StudyMetrics>({ // Added studyMetrics state
    timeSpent: 0,
    correctAnswers: 0,
    totalQuestions: 0,
    skillPerformance: {},
    difficultyPerformance: {}
  });


  // Practice mode configurations
  const practiceConfigs = {
    quick: { duration: 600, passages: 1, xp: 1 },    // 10 minutes
    full: { duration: 5400, passages: practiceContent.full.passages.length, xp: 2 },    // 90 minutes
    targeted: { duration: 1800, passages: 3, xp: 1.5 } // 30 minutes
  };

  // Timer effect
  useEffect(() => {
    let interval = null;
    if (practiceStarted && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining(time => time - 1);
      }, 1000);
    } else if (practiceStarted && timeRemaining === 0) {
      handleFinishTest();
    }
    return () => clearInterval(interval);
  }, [practiceStarted, timeRemaining]);

  const startPractice = (selectedMode) => {
    setMode(selectedMode);
    setTimeRemaining(practiceConfigs[selectedMode].duration);
    setPracticeStarted(true);
    setStartTime(Date.now()); // Set startTime when practice starts
    setCurrentPassage(practiceContent[selectedMode].passages[0]); // Updated passage selection
    setPassages(practiceContent[selectedMode].passages); // Updated passage setting
    setAnsweredQuestions({});
    setHasSubmitted(false);
    setStudyMetrics({ // Reset studyMetrics
      timeSpent: 0,
      correctAnswers: 0,
      totalQuestions: 0,
      skillPerformance: {},
      difficultyPerformance: {}
    });
  };

  const handleAnswer = (answerIndex: number) => { // Updated handleAnswer function
    setSelectedAnswer(answerIndex);
    setAnsweredQuestions(prev => ({
      ...prev,
      [currentPassage.questions[currentQuestion].id]: answerIndex
    }));
  
    const isCorrect = answerIndex === passages[currentPassageIndex].questions[currentQuestion].correct;
  
    // Update running totals
    setProgress(prev => {
      const newTotalAnswered = prev.totalAnswered + 1;
      const newTotalCorrect = prev.totalCorrect + (isCorrect ? 1 : 0);
      return {
        ...prev,
        totalAnswered: newTotalAnswered,
        totalCorrect: newTotalCorrect,
        accuracy: (newTotalCorrect / newTotalAnswered) * 100
      };
    });
  
    // Award XP if answer is correct
    if (isCorrect) {
      addXP(XP_RULES.CORRECT_ANSWER);
    }
  };

  const allQuestionsAnswered = () => {
    return Object.keys(answeredQuestions).length === (passages[currentPassageIndex]?.questions?.length || 0);
  };

  const handleNextQuestion = () => {
    const currentPassage = passages[currentPassageIndex];
    if (!currentPassage) return;

    if (currentQuestion < (currentPassage.questions?.length || 0) - 1) {
      setCurrentQuestion(prev => prev + 1);
      setShowExplanation(false);
      setSelectedAnswer(null);
    } else if (currentPassageIndex < passages.length - 1) {
      setCurrentPassageIndex(prev => prev + 1);
      setCurrentQuestion(0);
      setShowExplanation(false);
      setSelectedAnswer(null);
    }
  };

  const handlePreviousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
      setShowExplanation(false);
      setSelectedAnswer(null);
    } else if (currentPassageIndex > 0) {
      setCurrentPassageIndex(prev => prev - 1);
      setCurrentQuestion(passages[currentPassageIndex - 1].questions.length - 1);
      setShowExplanation(false);
      setSelectedAnswer(null);
    }
  };

  const handleFinishTest = () => {
    if (hasSubmitted || !allQuestionsAnswered()) {
      return;
    }

    const currentPassage = passages[currentPassageIndex];
    if (!currentPassage) return;

    // If there are more passages, don't mark as submitted yet
    if (currentPassageIndex < passages.length - 1) {
      setCurrentPassageIndex(prev => prev + 1);
      setCurrentQuestion(0);
      setShowExplanation(false);
      setSelectedAnswer(null);
      setAnsweredQuestions({});
      return;
    }

    const totalQuestions = passages.reduce((sum, passage) => sum + passage.questions.length, 0); // Updated accuracy calculation
    const correctAnswers = Object.entries(answeredQuestions).reduce((sum, [questionId, answer]) => {
      // Find the question across all passages
      const question = passages.flatMap(p => p.questions).find(q => q.id === parseInt(questionId));
      return sum + (question && answer === question.correct ? 1 : 0);
    }, 0);

    const accuracy = (correctAnswers / totalQuestions) * 100; // Updated accuracy calculation
    const timeSpent = Date.now() - startTime; // Calculate time spent
    const xpGained = Math.floor(accuracy);
    const newStreak = allQuestionsAnswered() ? progress.streak + 1 : 0;
    const streakBonus = newStreak * 10;
    const totalXp = xpGained + streakBonus;

    const newProgress = {
      accuracy,
      streak: newStreak,
      masteryPoints: Math.min(100, progress.masteryPoints + Math.floor(accuracy / 2)),
      xp: progress.xp + totalXp,
      level: Math.floor((progress.xp + totalXp) / 1000) + 1,
      totalAnswered: totalQuestions,
      totalCorrect: correctAnswers
    };

    const skillPerformance = getProgressBySkill(answeredQuestions, passages[currentPassageIndex].questions);

    setProgress(newProgress);
    setHasSubmitted(true);
    setStudyMetrics({ // Update studyMetrics
      timeSpent: timeSpent,
      correctAnswers: correctAnswers,
      totalQuestions: totalQuestions,
      skillPerformance: skillPerformance,
      difficultyPerformance: {} // Add difficultyPerformance calculation if needed
    });

    if (onUpdateStats) {
      onUpdateStats({
        xp: totalXp, // Updated XP calculation
        correct: correctAnswers,
        total: totalQuestions
      });
    }

    console.log('CARS Practice - XP Update:', { // Added console.log statements
      xpGained,
      streakBonus,
      totalXp,
      correctAnswers,
      totalQuestions
    });
  };

  const renderModeSelector = () => (
    <div className="space-y-4">
      {Object.entries(practiceConfigs).map(([key, config]) => (
        <button
          key={key}
          onClick={() => startPractice(key)}
          className="w-full p-6 bg-[#1e2330] hover:bg-[#252b3b] transition-colors rounded-xl text-left"
        >
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-xl font-bold mb-2">
                {key === 'quick' ? 'Quick Practice' :
                 key === 'full' ? 'Full Section' :
                 'Targeted Practice'}
              </h3>
              <div className="text-gray-400">
                {config.duration / 60} minutes • {config.passages} passage{config.passages > 1 ? 's' : ''}
              </div>
            </div>
            <div className="text-yellow-400 font-medium">
              {config.xp}x XP
            </div>
          </div>
        </button>
      ))}
    </div>
  );

  const renderPassage = () => {
    if (!passages[currentPassageIndex]) {
      return <div>No passage available</div>;
    }
    return (
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div className="flex items-center space-x-4">
            <Book className="h-6 w-6" />
            <h2 className="text-xl font-bold">Passage {passages[currentPassageIndex].id}</h2>
          </div>
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5" />
            <span className="text-lg">
              {Math.floor(timeRemaining / 60)}:{String(timeRemaining % 60).padStart(2, '0')}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-gray-400">Passage</span>
          <span className="font-bold">{currentPassageIndex + 1}</span>
          <span className="text-gray-400">of {passages.length}</span>
          <span className="mx-2">•</span>
          <span className="text-gray-400">Question</span>
          <span className="font-bold">{currentQuestion + 1}</span>
          <span className="text-gray-400">of {passages[currentPassageIndex].questions.length}</span>
        </div>
        <div className="flex space-x-4">
          <Button variant="secondary" className="bg-[#1e2330] hover:bg-[#252b3b]">
            <Highlighter className="h-4 w-4 mr-2" />
            Highlight Tool
          </Button>
          <Button variant="secondary" className="bg-[#1e2330] hover:bg-[#252b3b]">
            <Plus className="h-4 w-4 mr-2" />
            Add Note
          </Button>
        </div>

        <Card className="bg-[#1e2330] border-0">
          <div className="p-6">
            <h3 className="text-xl font-bold mb-2">{passages[currentPassageIndex]?.title}</h3>
            <p className="text-gray-400 mb-4">Source: {passages[currentPassageIndex]?.source}</p>
            <p className="text-gray-200 leading-relaxed">{passages[currentPassageIndex]?.content}</p>
          </div>
        </Card>

        <div className="grid grid-cols-2 gap-6">
          <Card className="bg-[#1e2330] border-0">
            <div className="p-4">
              <h4 className="font-medium mb-3">Key Terms</h4>
              <div className="flex flex-wrap gap-2">
                {passages[currentPassageIndex].keyTerms.map(term => (
                  <span key={term} className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                    {term}
                  </span>
                ))}
              </div>
            </div>
          </Card>

          <Card className="bg-[#1e2330] border-0">
            <div className="p-4">
              <h4 className="font-medium mb-3">Main Themes</h4>
              <div className="flex flex-wrap gap-2">
                {passages[currentPassageIndex].mainThemes.map(theme => (
                  <span key={theme} className="px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm">
                    {theme}
                  </span>
                ))}
              </div>
            </div>
          </Card>
        </div>

        <Card className="bg-[#1e2330] border-0">
          <div className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-lg font-medium">Question {currentQuestion + 1}</h4>
              <span className="text-sm text-gray-400">Main Idea</span>
            </div>

            <p className="text-lg mb-6">{passages[currentPassageIndex]?.questions?.[currentQuestion]?.text}</p>

            <div className="space-y-3">
              {passages[currentPassageIndex]?.questions?.[currentQuestion]?.options.map((option, index) => (
                <button
                  key={index}
                  onClick={() => handleAnswer(index)}
                  className={`w-full text-left p-4 rounded-lg transition-colors ${
                    selectedAnswer === index 
                      ? index === passages[currentPassageIndex].questions[currentQuestion].correct
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-red-500/20 text-red-400'
                      : 'bg-[#252b3b] hover:bg-[#2a3142]'
                  }`}
                >
                  {String.fromCharCode(65 + index)}. {option}
                </button>
              ))}
            </div>

            <div className="flex justify-between mt-6">
              <Button 
                variant="outline" 
                className="bg-[#252b3b]"
                onClick={handlePreviousQuestion}
                disabled={currentQuestion === 0 && currentPassageIndex === 0}
              >
                <ChevronLeft className="mr-2 h-4 w-4" />
                Previous
              </Button>
              <Button 
                variant="outline" 
                className="bg-[#252b3b]"
                onClick={() => setShowExplanation(!showExplanation)}
              >
                {showExplanation ? 'Hide' : 'Show'} Explanation
              </Button>
              {currentQuestion === passages[currentPassageIndex].questions.length - 1 ? (
                currentPassageIndex < passages.length - 1 ? (
                  <Button
                    onClick={() => {
                      setCurrentPassageIndex(prev => prev + 1);
                      setCurrentQuestion(0);
                      setShowExplanation(false);
                      setSelectedAnswer(null);
                      setAnsweredQuestions({});
                    }}
                    className="bg-primary hover:bg-primary/90"
                  >
                    Next Passage
                  </Button>
                ) : (
                  <Button
                    onClick={handleFinishTest}
                    className="bg-primary hover:bg-primary/90"
                    disabled={!allQuestionsAnswered() || hasSubmitted}
                  >
                    {hasSubmitted ? 'Completed' : 'Finish'}
                  </Button>
                )
              ) : (
                <Button 
                  variant="outline" 
                  className="bg-[#252b3b]"
                  onClick={handleNextQuestion}
                  disabled={selectedAnswer === null}
                >
                  Next
                  <ChevronRight className="ml-2 h-4 w-4" />
                </Button>
              )}
            </div>

            {showExplanation && (
              <div className="mt-6 p-4 bg-[#252b3b] rounded-lg">
                <h5 className="font-medium mb-2">Explanation:</h5>
                <p className="text-gray-300">
                  {passages[currentPassageIndex].questions[currentQuestion].explanation}
                </p>
              </div>
            )}
          </div>
        </Card>
      </div>
    );
  };

  const renderProgress = () => (
    <div className="mt-8 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Your Progress</h2>
        <div className="flex items-center space-x-4">
          <span className="text-yellow-400">{progress.xp} XP</span>
          <span className="text-blue-400">Level {progress.level}</span>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <div className="text-gray-400 mb-2">Accuracy</div>
          <div className="text-3xl font-bold mb-2">{progress.accuracy.toFixed(0)}%</div>
          <Progress 
            value={progress.accuracy}
            className="h-2 bg-gray-700"
          />
        </div>

        <div>
          <div className="text-gray-400 mb-2">Current Streak</div>
          <div className="text-3xl font-bold">{progress.streak}</div>
          <div className="text-green-400 text-sm mt-1">
            +{progress.streak * 10} XP Bonus
          </div>
          {!allQuestionsAnswered() && (
            <div className="text-yellow-400 text-sm mt-1">
              Complete all questions to increase streak
            </div>
          )}
        </div>

        <div>
          <div className="text-gray-400 mb-2">Questions Completed</div>
          <div className="text-3xl font-bold mb-2">
            {Object.keys(answeredQuestions).length}/{passages[currentPassageIndex]?.questions?.length || 0}
          </div>
          <Progress 
            value={(Object.keys(answeredQuestions).length / (passages[currentPassageIndex]?.questions?.length || 1)) * 100}
            className="h-2 bg-gray-700"
          />
        </div>

        <div>
          <div className="text-gray-400 mb-2">Mastery Progress</div>
          <div className="text-3xl font-bold mb-2">{progress.masteryPoints}/100</div>
          <Progress 
            value={progress.masteryPoints}
            className="h-2 bg-gray-700"
          />
        </div>
      </div>
    </div>
  );


  return (
    <div className="min-h-screen bg-[#111827] text-white p-6">
      <div className="max-w-4xl mx-auto">
        {practiceStarted && (
          <Button
            onClick={onExitToMenu}
            variant="ghost"
            className="mb-6 bg-[#1e2330]/80 hover:bg-[#252b3b] text-white"
          >
            <Home className="mr-2 h-4 w-4" />
            Exit to Menu
          </Button>
        )}

        {!practiceStarted && renderModeSelector()}
        {practiceStarted && renderPassage()}
        {renderProgress()}
      </div>
    </div>
  );
}

