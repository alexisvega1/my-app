import { NextResponse } from 'next/server'
import prisma from '@/lib/prisma'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const deck = searchParams.get('deck')
  const page = parseInt(searchParams.get('page') || '1')
  const limit = 100 // Number of flashcards per page

  if (!deck) {
    return NextResponse.json({ error: 'Deck parameter is required' }, { status: 400 })
  }

  try {
    const flashcards = await prisma.flashcard.findMany({
      where: { deckId: deck },
      take: limit,
      skip: (page - 1) * limit,
    })

    const totalCount = await prisma.flashcard.count({
      where: { deckId: deck },
    })

    return NextResponse.json({
      flashcards,
      totalPages: Math.ceil(totalCount / limit),
      currentPage: page,
    })
  } catch (error) {
    console.error('Error fetching flashcards:', error)
    return NextResponse.json({ error: 'Error fetching flashcards' }, { status: 500 })
  }
}

