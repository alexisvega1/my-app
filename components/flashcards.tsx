"use client"

import { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { ChevronLeft, ChevronRight, BookOpen } from 'lucide-react'
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { motion } from "framer-motion"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

const decks = [
  { id: 'js2048', name: 'Jacksparrow2048 MCAT Anki UPDATED' },
  { id: 'miledown', name: "MileDown's MCAT Anki Deck" },
  { id: 'abdullah', name: "Abdullah's Deck" },
  { id: 'coffin', name: 'Coffin milesdown' },
  { id: 'pankow', name: 'Mr_Pankow_P_S Deck' }
]

export function Flashcards() {
  const [selectedDeck, setSelectedDeck] = useState('')
  const [currentCard, setCurrentCard] = useState(0)
  const [showAnswer, setShowAnswer] = useState(false)

  const { data: flashcards, isLoading, error } = useQuery(
    ['flashcards', selectedDeck],
    () => fetch(`/api/flashcards?deck=${selectedDeck}`).then(res => res.json()),
    { enabled: !!selectedDeck }
  )

  const handleDeckChange = (deckId: string) => {
    setSelectedDeck(deckId)
    setCurrentCard(0)
    setShowAnswer(false)
  }

  const nextCard = () => {
    if (flashcards && currentCard < flashcards.length - 1) {
      setCurrentCard(prev => prev + 1)
      setShowAnswer(false)
    }
  }

  const prevCard = () => {
    if (currentCard > 0) {
      setCurrentCard(prev => prev - 1)
      setShowAnswer(false)
    }
  }

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error loading flashcards</div>

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-6">
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <BookOpen className="text-gray-400" size={20} />
          <Select value={selectedDeck} onValueChange={handleDeckChange}>
            <SelectTrigger className="w-full bg-gray-700 border-gray-600 text-white">
              <SelectValue placeholder="Select MCAT Deck" />
            </SelectTrigger>
            <SelectContent className="bg-gray-700 border-gray-600">
              {decks.map((deck) => (
                <SelectItem 
                  key={deck.id} 
                  value={deck.id}
                  className="text-white hover:bg-gray-600 focus:bg-gray-600"
                >
                  {deck.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {flashcards && flashcards.length > 0 && (
        <>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card 
              className="bg-gray-800 rounded-xl shadow-lg h-96 cursor-pointer transition-transform transform hover:scale-105 border-gray-700"
              onClick={() => setShowAnswer(!showAnswer)}
            >
              <CardContent className="h-full flex flex-col justify-between p-8">
                <div className="text-center">
                  {!showAnswer ? (
                    <h2 className="text-xl font-medium text-white">{flashcards[currentCard].question}</h2>
                  ) : (
                    <p className="text-lg text-white">{flashcards[currentCard].answer}</p>
                  )}
                </div>
                <div className="text-sm text-gray-400 text-center">
                  Click to {showAnswer ? 'hide' : 'show'} answer
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <div className="flex justify-between mt-6">
            <Button
              onClick={prevCard}
              variant="outline"
              className="flex items-center space-x-2 bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
              disabled={currentCard === 0}
            >
              <ChevronLeft size={20} />
              <span>Previous</span>
            </Button>
            <div className="text-gray-400 flex items-center space-x-2 text-xs">
              <Progress 
                value={((currentCard + 1) / flashcards.length) * 100} 
                className={cn("w-24 h-1 bg-gray-700", "data-[value]:bg-primary")}
              />
              <span>
                Card {currentCard + 1} of {flashcards.length}
              </span>
            </div>
            <Button
              onClick={nextCard}
              variant="outline"
              className="flex items-center space-x-2 bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
              disabled={currentCard === flashcards.length - 1}
            >
              <span>Next</span>
              <ChevronRight size={20} />
            </Button>
          </div>
        </>
      )}
    </div>
  )
}

