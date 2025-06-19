#!/usr/bin/env python3
"""
Project Manager
==============
Handles backlog management and ticket creation for the multi-agent system.
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ProjectManager:
    """Manages the project backlog and ticket creation."""
    
    def __init__(self, backlog_file: str = "backlog.json"):
        self.backlog_file = backlog_file
        self.backlog = self._load_backlog()
    
    def _load_backlog(self) -> List[Dict[str, Any]]:
        """Load the backlog from file."""
        try:
            with open(self.backlog_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"Creating new backlog file: {self.backlog_file}")
            return []
        except json.JSONDecodeError:
            logger.warning(f"Corrupted backlog file, starting fresh: {self.backlog_file}")
            return []
    
    def _save_backlog(self):
        """Save the backlog to file."""
        try:
            with open(self.backlog_file, 'w') as f:
                json.dump(self.backlog, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backlog: {e}")
    
    def add_to_backlog(self, description: str, priority: str = "medium") -> Dict[str, Any]:
        """Add a new ticket to the backlog."""
        ticket = {
            "id": str(uuid.uuid4()),
            "description": description,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.backlog.append(ticket)
        self._save_backlog()
        
        logger.info(f"Added ticket {ticket['id']} to backlog")
        return ticket
    
    def get_pending_tickets(self) -> List[Dict[str, Any]]:
        """Get all pending tickets."""
        return [ticket for ticket in self.backlog if ticket['status'] == 'pending']
    
    def update_ticket_status(self, ticket_id: str, status: str):
        """Update the status of a ticket."""
        for ticket in self.backlog:
            if ticket['id'] == ticket_id:
                ticket['status'] = status
                ticket['updated_at'] = datetime.now().isoformat()
                self._save_backlog()
                logger.info(f"Updated ticket {ticket_id} status to {status}")
                return
        
        logger.warning(f"Ticket {ticket_id} not found")
    
    def get_backlog_stats(self) -> Dict[str, Any]:
        """Get backlog statistics."""
        total = len(self.backlog)
        pending = len([t for t in self.backlog if t['status'] == 'pending'])
        completed = len([t for t in self.backlog if t['status'] == 'completed'])
        
        return {
            "total": total,
            "pending": pending,
            "completed": completed
        }

def load_backlog() -> List[Dict[str, Any]]:
    """Utility function to load the backlog."""
    pm = ProjectManager()
    return pm.backlog 