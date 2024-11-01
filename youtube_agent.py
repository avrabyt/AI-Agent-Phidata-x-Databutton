import os
import databutton as db
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from phi.agent import Agent
from phi.tools.youtube_tools import YouTubeTools
import re

router = APIRouter()

class VideoRequest(BaseModel):
    url: str

class VideoMetadata(BaseModel):
    title: str
    thumbnail_url: str
    video_id: str
    transcript: str

class QuestionRequest(BaseModel):
    video_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    # Regular expressions for different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([\w-]{11})(?:\?|&|$)',  # Standard and shortened URLs
        r'^([\w-]{11})$'  # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL")

@router.post("/process_video_url")
def process_video_url(request: VideoRequest) -> VideoMetadata:
    """Process YouTube URL and return video metadata and transcript"""
    try:
        # Extract video ID
        video_id = extract_video_id(request.url)
        
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_list])
        
        # Use video ID to construct thumbnail URL
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        return VideoMetadata(
            title=f"YouTube Video {video_id}",  # Simple title using video ID
            thumbnail_url=thumbnail_url,
            video_id=video_id,
            transcript=transcript_text
        )
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

@router.post("/ask-question")
def ask_question(request: QuestionRequest) -> QuestionResponse:
    """Answer questions about the video using the YouTube agent"""
    try:
        # Get OpenAI API key from secrets
        openai_api_key = db.secrets.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in secrets")
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize YouTube agent
        agent = Agent(
            tools=[YouTubeTools()],
            show_tool_calls=False,
            description="You are a YouTube agent. Obtain the captions of a YouTube video and answer questions. Provide concise, focused answers and use markdown formatting for better readability."
        )
        
        # Format the video URL and question
        video_url = f"https://www.youtube.com/watch?v={request.video_id}"
        prompt = f"{request.question}\nVideo: {video_url}"
        
        # Get agent's response
        response = agent.run(prompt)
        # Extract just the content from the response
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return QuestionResponse(answer=answer)
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        raise
