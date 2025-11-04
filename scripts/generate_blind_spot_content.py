#!/usr/bin/env python3
"""
Generate content suggestions for blind spots.

Automatically creates:
- Tweet hooks
- Thread outlines
- Blog skeletons
Based on blind spot type and private content.
"""
from typing import List, Dict, Optional
from datetime import datetime

def generate_tweet_hook(blind_spot_type: str, private_content_summary: str, 
                        max_similarity: float, cluster_id: int) -> str:
    """Generate a tweet hook based on blind spot type."""
    
    if blind_spot_type == 'reactivation_gap':
        hooks = [
            "Most important ideas only make sense after they play out. Here's my rule for publishing in uncertainty.",
            "Revisiting a thread from 2022 with 2025 clarity: sometimes you need to sit with ideas before they're ready.",
            "What I learned: ideas that feel obvious now were impossible to explain 2 years ago. Here's why.",
        ]
        return hooks[0]  # Default to first, could randomize
    
    elif blind_spot_type == 'burst_no_collapse':
        hooks = [
            "What looks like chaos is a phase transition. Outliers on {date} → synthesis on {date}. Here's how to engineer it.",
            "The geometry of innovation is a timing problem: you can't accelerate faster than your team's temporal bandwidth.",
            "Multiple conversations converged but nothing published. Here's the gap I found and how to close it.",
        ]
        return hooks[0].format(date="10/29")  # Could extract from context
    
    elif blind_spot_type == 'channel_mismatch':
        hooks = [
            "Some ideas need more space than a tweet. Here's what I learned about {topic} and why it needed a blog post.",
            "Research-dense ideas don't fit in 280 characters. Thread + blog for {topic}.",
            "Tried to explain {topic} in a tweet. Realized it needs 1200 words. Here's why.",
        ]
        # Extract topic from summary
        topic = private_content_summary.split()[0] if private_content_summary else "this"
        return hooks[0].format(topic=topic)
    
    elif blind_spot_type == 'new_topic':
        hooks = [
            "New territory: exploring {topic}. Here's what I've learned so far.",
            "First time writing about {topic}. Here's why it matters now.",
        ]
        topic = private_content_summary.split()[0] if private_content_summary else "this"
        return hooks[0].format(topic=topic)
    
    else:
        # Default hook
        return f"Working on Cluster {cluster_id}. Here's what I'm thinking."

def generate_thread_outline(blind_spot_type: str, private_content_summary: str,
                           aligned_public_clusters: str) -> List[str]:
    """Generate a 5-bullet thread outline."""
    
    if blind_spot_type == 'reactivation_gap':
        outline = [
            "1. Problem: Why ideas need time to mature",
            "2. Network relativity lens: how concepts evolve",
            "3. Teacher/class time model: the temporal bandwidth constraint",
            "4. Operational metric: Insight Velocity (IV)",
            "5. Call for examples: what's your longest incubation period?"
        ]
    elif blind_spot_type == 'burst_no_collapse':
        outline = [
            "1. What looks like chaos is a phase transition",
            "2. Outliers → synthesis window (the signal)",
            "3. Why some bursts don't collapse (the gap)",
            "4. How to engineer the collapse (forcing functions)",
            "5. Operational playbook: predict your next synthesis window"
        ]
    elif blind_spot_type == 'channel_mismatch':
        outline = [
            "1. Some ideas need more space than tweets",
            "2. Research-dense → blog/whitepaper format",
            "3. Why {topic} needed long-form",
            "4. Format decision tree: tweet vs thread vs blog",
            "5. CTA: Link to full post"
        ]
    else:
        outline = [
            "1. The insight",
            "2. Why it matters",
            "3. Practical implications",
            "4. Examples",
            "5. Call to action"
        ]
    
    return outline

def generate_blog_outline(blind_spot_type: str, private_content_summary: str,
                         word_count: int = 1200) -> Dict[str, any]:
    """Generate a blog post outline with sections."""
    
    outline = {
        'title': f"Understanding {private_content_summary[:50]}...",
        'target_word_count': word_count,
        'sections': []
    }
    
    if blind_spot_type == 'reactivation_gap':
        outline['sections'] = [
            {
                'heading': 'Introduction: The Long Incubation',
                'words': 200,
                'content': 'Hook: Most important ideas only make sense after they play out. Personal story of revisiting 2022 threads with 2025 clarity.'
            },
            {
                'heading': 'The Network Relativity Framework',
                'words': 300,
                'content': 'Explain network relativity lens and how concepts evolve over time. Teacher/class temporal bandwidth model.'
            },
            {
                'heading': 'Measuring Insight Velocity',
                'words': 300,
                'content': 'Operational metric: IV. How to track idea incubation and synthesis. Include your teacher-student-AI model.'
            },
            {
                'heading': 'The Reactivation Pattern',
                'words': 250,
                'content': 'Analysis of before/after: 2022 threads vs 2025 model. What changed, what stayed the same.'
            },
            {
                'heading': 'Conclusion: Publishing in Uncertainty',
                'words': 150,
                'content': 'Rule for publishing when ideas aren\'t fully formed. Call for examples from readers.'
            }
        ]
    elif blind_spot_type == 'burst_no_collapse':
        outline['sections'] = [
            {
                'heading': 'Phase Transitions in Creative Work',
                'words': 200,
                'content': 'What looks like chaos is actually a phase transition. Outliers → synthesis window.'
            },
            {
                'heading': 'The Signal: Multiple Private Conversations',
                'words': 300,
                'content': 'Why burst of private activity signals incoming synthesis. The patterns that predict collapse.'
            },
            {
                'heading': 'The Gap: Why Some Bursts Don\'t Collapse',
                'words': 300,
                'content': 'What prevents synthesis from happening. Missing forcing functions, lack of external triggers.'
            },
            {
                'heading': 'Engineering the Collapse',
                'words': 300,
                'content': 'How to force synthesis: publishing triggers, collaborator feedback, deadline pressure.'
            },
            {
                'heading': 'Operational Playbook',
                'words': 100,
                'content': 'CTA: DM for TrustOps audit. I\'ll run your last 7 days and predict your next synthesis window.'
            }
        ]
    elif blind_spot_type == 'channel_mismatch':
        outline['sections'] = [
            {
                'heading': 'The Format Problem',
                'words': 200,
                'content': 'Some ideas need more space than tweets. Why research-dense content needs long-form.'
            },
            {
                'heading': 'Why {topic} Needed a Blog Post',
                'words': 400,
                'content': 'Deep dive into the topic. Why it couldn\'t fit in 280 characters. Include diagrams if relevant.'
            },
            {
                'heading': 'The Format Decision Tree',
                'words': 300,
                'content': 'When to use: tweet vs thread vs blog vs whitepaper. Decision framework.'
            },
            {
                'heading': 'Examples and Case Studies',
                'words': 200,
                'content': 'Examples of ideas that needed different formats. Lessons learned.'
            },
            {
                'heading': 'Conclusion',
                'words': 100,
                'content': 'Key takeaway and call to action.'
            }
        ]
    else:
        outline['sections'] = [
            {
                'heading': 'Introduction',
                'words': 200,
                'content': 'Hook and context'
            },
            {
                'heading': 'Main Argument',
                'words': 600,
                'content': 'Core insights and analysis'
            },
            {
                'heading': 'Practical Implications',
                'words': 300,
                'content': 'How to apply these insights'
            },
            {
                'heading': 'Conclusion',
                'words': 100,
                'content': 'Summary and next steps'
            }
        ]
    
    return outline

def generate_content_suggestions(blind_spot_row: Dict, private_content_summary: str = "") -> Dict:
    """Generate complete content suggestions for a blind spot."""
    
    blind_spot_type = blind_spot_row.get('blind_spot_type', 'unknown')
    max_similarity = blind_spot_row.get('max_similarity', 0.0)
    cluster_id = blind_spot_row.get('private_cluster', 0)
    aligned_clusters = blind_spot_row.get('aligned_public_clusters', '')
    
    return {
        'tweet_hook': generate_tweet_hook(blind_spot_type, private_content_summary, max_similarity, cluster_id),
        'thread_outline': generate_thread_outline(blind_spot_type, private_content_summary, aligned_clusters),
        'blog_outline': generate_blog_outline(blind_spot_type, private_content_summary),
        'recommended_format': _get_recommended_format(blind_spot_type),
        'action_items': _get_action_items(blind_spot_type)
    }

def _get_recommended_format(blind_spot_type: str) -> str:
    """Get recommended content format based on type."""
    if blind_spot_type == 'reactivation_gap':
        return "Thread + blog post (link to historical tweets for continuity)"
    elif blind_spot_type == 'burst_no_collapse':
        return "Force collapse: turn private files into one artifact (thread + image/carousel)"
    elif blind_spot_type == 'channel_mismatch':
        return "Blog/whitepaper or long tweet thread + image (don't squeeze into single tweet)"
    else:
        return "Thread or blog post"

def _get_action_items(blind_spot_type: str) -> List[str]:
    """Get action items based on blind spot type."""
    if blind_spot_type == 'reactivation_gap':
        return [
            "Publish an updated take (thread + blog)",
            "Link to historical tweets for continuity",
            "Show before/after comparison (2022 vs 2025)"
        ]
    elif blind_spot_type == 'burst_no_collapse':
        return [
            "Force a collapse: turn private files into one artifact",
            "Create synthesis artifact (7-slide carousel or blog post)",
            "Include Insight Thermodynamics charts if available"
        ]
    elif blind_spot_type == 'channel_mismatch':
        return [
            "Use blog/whitepaper or long tweet thread",
            "Include diagrams or images if relevant",
            "Don't try to squeeze into a single tweet"
        ]
    else:
        return [
            "Generate content from private exploration",
            "Match format to content depth",
            "Include examples and practical implications"
        ]
