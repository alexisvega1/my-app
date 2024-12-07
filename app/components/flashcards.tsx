import { useState, useEffect } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { offlineStorage } from '../utils/offlineStorage'
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"

const defaultFlashcards = [
  { question: "What is an amino acid?", answer: "An amino acid is a molecule that contains an amino group (-NH2) and a carboxyl group (-COOH)." },
  { question: "What is a side chain?", answer: "A side chain, or R group, is the part of the amino acid that defines and specifies the various amino acids." },
  { question: "What are the proteinogenic amino acids?", answer: "Proteinogenic amino acids are the 20 alpha amino acids encoded by the human genetic code. These are the only amino acids with which you need to have a memorized familiarity." },
  { question: "With the exception of glycine, all amino acids are [...]", answer: "With the exception of glycine, all amino acids are chiral" },
  { question: "All chiral amino acids in eukaryotes are [L- or D- amino acids?]", answer: "All chiral amino acids in eukaryotes are L" },
  { question: "What are Glycine's structure, character, three letter abbreviation, and one letter abbreviation?", answer: "Glycine, Gly, or G is a non-polar aliphatic amino acid. Also unique in that it is non-chiral!" },
  { question: "What are Valine's structure, character, three letter abbreviation, and one letter abbreviation?", answer: "Valine, Val, or V is a non-polar, aliphatic amino acid." },
  { question: "What are Leucine's structure, character, three letter abbreviation, and one letter abbreviation?", answer: "Leucine, Leu, or L is a non-polar aliphatic amino acid." },
  { question: "What are Alanine's structure, character, three letter abbreviation, and one letter abbreviation?", answer: "Alanine, Ala, or A is a non-polar aliphatic amino acid." },
  { question: "What are Isoleucine's structure, character, three letter abbreviation, and one letter abbreviation?", answer: "Isoleucine, Ile, or I is a non-polar, aliphatic amino acid." },
  // ... (include all the flashcards provided by the user)
]

export function Flashcards() {
  const [currentCard, setCurrentCard] = useState(0)
  const [showAnswer, setShowAnswer] = useState(false)
  const [flashcards, setFlashcards] = useState(defaultFlashcards)

  useEffect(() => {
    async function loadFlashcards() {
      const storedFlashcards = await offlineStorage.getItem('flashcards')
      if (storedFlashcards) {
        setFlashcards(storedFlashcards)
      }
    }
    loadFlashcards()
  }, [])

  const saveFlashcards = async (updatedFlashcards) => {
    await offlineStorage.setItem('flashcards', updatedFlashcards)
    setFlashcards(updatedFlashcards)
  }

  const nextCard = () => {
    setCurrentCard((prev) => (prev + 1) % flashcards.length)
    setShowAnswer(false)
  }

  const prevCard = () => {
    setCurrentCard((prev) => (prev - 1 + flashcards.length) % flashcards.length)
    setShowAnswer(false)
  }

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card 
          className="bg-white dark:bg-gray-800 rounded-xl shadow-lg h-96 cursor-pointer transition-transform transform hover:scale-105"
          onClick={() => setShowAnswer(!showAnswer)}
        >
          <CardContent className="h-full flex flex-col justify-between p-8">
            <div className="text-center">
              {!showAnswer ? (
                <h2 className="text-xl font-medium">{flashcards[currentCard]?.question}</h2>
              ) : (
                <p className="text-lg">{flashcards[currentCard]?.answer}</p>
              )}
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-400 text-center">
              Click to {showAnswer ? 'hide' : 'show'} answer
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="flex justify-between mt-6">
        <Button
          onClick={prevCard}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <ChevronLeft size={20} />
          <span>Previous</span>
        </Button>
        <div className="text-gray-500 dark:text-gray-400">
          Card {currentCard + 1} of {flashcards.length}
        </div>
        <Button
          onClick={nextCard}
          variant="outline"
          className="flex items-center space-x-2"
        >
          <span>Next</span>
          <ChevronRight size={20} />
        </Button>
      </div>
    </div>
  )
}

