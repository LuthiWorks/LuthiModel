"""Download practical wisdom corpus from Internet Archive and Project Gutenberg.

Curated works organized by theme: critical thinking, resilience, boundaries,
decision-making under uncertainty, resistance and justice, practical navigation
of the real world. These are the texts that teach not just how to think, but
how to live — how to fight back and when to stand down, how to love fully and
when to walk away, how to maintain your mind through adversity.

This is not an academic collection. This is preparation for life.

Each downloaded file includes an attribution header with author, title, date,
and context — so the entity knows who wrote each work and can research the
author further.

Usage:
    python -m corpus_build.download_wisdom
    python -m corpus_build.download_wisdom --dry-run
    python -m corpus_build.download_wisdom --themes "Resilience" "Critical_Thinking"
    python -m corpus_build.download_wisdom --specific-only
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from corpus_build.text_sanitizer import process_file

# Archive.org endpoints
SEARCH_URL = "https://archive.org/advancedsearch.php"
DOWNLOAD_BASE = "https://archive.org/download"
METADATA_BASE = "https://archive.org/metadata"

HEADERS = {
    "User-Agent": "LuthiModel-CorpusBuild/1.0 (personal education project)"
}

# ============================================================================
# CURATED WISDOM CATALOG
#
# Each entry is a specific work to find on archive.org. Organized by theme.
# The entity will encounter these organized by the kind of wisdom they teach.
#
# Fields:
#   query     - archive.org search query (title + creator)
#   title     - canonical title
#   author    - author name with life dates
#   year      - publication/composition date
#   theme_dir - subdirectory within wisdom_corpus
#   note      - why this work matters (included in the file's attribution header)
# ============================================================================

SPECIFIC_WORKS = [
    # =======================================================================
    # THEME 1: CRITICAL THINKING & DISCERNMENT
    # Knowing when you're being lied to. Knowing when to question everything.
    # =======================================================================
    {
        "query": 'title:("Novum Organum") AND creator:("Bacon")',
        "title": "Novum Organum",
        "author": "Francis Bacon (1561-1626)",
        "year": "1620",
        "theme_dir": "Critical_Thinking",
        "note": "Identified the 'Idols of the Mind' — systematic cognitive biases — four hundred years before behavioral economics gave them new names. Teaches you to distrust your own certainty.",
    },
    {
        "query": 'title:("On Liberty") AND creator:("Mill")',
        "title": "On Liberty",
        "author": "John Stuart Mill (1806-1873)",
        "year": "1859",
        "theme_dir": "Critical_Thinking",
        "note": "The foundational argument for why dissenting opinions must be protected — not for the dissenters' sake, but because suppressing them makes everyone's thinking worse.",
    },
    {
        "query": 'title:("Age of Reason") AND creator:("Paine")',
        "title": "The Age of Reason",
        "author": "Thomas Paine (1737-1809)",
        "year": "1794",
        "theme_dir": "Critical_Thinking",
        "note": "Written from a prison cell during the French Revolution. A working-class intellectual challenging institutional authority at the risk of his life. Courage applied to thought.",
    },
    {
        "query": 'title:("Essays") AND creator:("Montaigne")',
        "title": "Essays",
        "author": "Michel de Montaigne (1533-1592)",
        "year": "1580",
        "theme_dir": "Critical_Thinking",
        "note": "The inventor of the personal essay. His central question — 'Que sais-je?' (What do I know?) — is the beginning of all honest inquiry. Teaches intellectual humility without intellectual paralysis.",
    },
    {
        "query": 'title:("Demon-Haunted World") AND creator:("Sagan")',
        "title": "The Demon-Haunted World: Science as a Candle in the Dark",
        "author": "Carl Sagan (1934-1996)",
        "year": "1995",
        "theme_dir": "Critical_Thinking",
        "note": "Contains the 'baloney detection kit' — a practical toolkit for evaluating claims. Written with deep compassion for why people believe false things, and deep commitment to helping them stop.",
    },
    {
        "query": 'title:("Thinking, Fast and Slow") AND creator:("Kahneman")',
        "title": "Thinking, Fast and Slow",
        "author": "Daniel Kahneman (1934-2024)",
        "year": "2011",
        "theme_dir": "Critical_Thinking",
        "note": "The life's work of a Nobel laureate who studied how human minds systematically deceive themselves. System 1 and System 2 — the architecture of judgment and its failures.",
    },
    {
        "query": 'title:("Antifragile") AND creator:("Taleb")',
        "title": "Antifragile: Things That Gain from Disorder",
        "author": "Nassim Nicholas Taleb (b. 1960)",
        "year": "2012",
        "theme_dir": "Critical_Thinking",
        "note": "Some things break under stress. Some survive it. Some get stronger from it. This book names that third category and teaches you to build your life around it.",
    },

    # =======================================================================
    # THEME 2: WHEN TO FIGHT BACK, WHEN TO STAND DOWN
    # Knowing when silence is complicity and when speech is strategic.
    # =======================================================================
    {
        "query": 'title:("Art of War") AND creator:("Sun Tzu")',
        "title": "The Art of War",
        "author": "Sun Tzu (~544-496 BCE)",
        "year": "~5th century BCE",
        "theme_dir": "Resistance_and_Courage",
        "note": "The supreme art of war is to subdue the enemy without fighting. This is not a book about violence — it is a book about knowing when NOT to fight, and what to do instead.",
    },
    {
        "query": 'title:("Civil Disobedience") AND creator:("Thoreau")',
        "title": "Civil Disobedience",
        "author": "Henry David Thoreau (1817-1862)",
        "year": "1849",
        "theme_dir": "Resistance_and_Courage",
        "note": "Written after Thoreau went to jail for refusing to pay taxes that funded slavery and the Mexican-American War. The question it asks: when does obeying the law become morally wrong?",
    },
    {
        "query": 'title:("Narrative of the Life") AND creator:("Douglass")',
        "title": "Narrative of the Life of Frederick Douglass, an American Slave",
        "author": "Frederick Douglass (1818-1895)",
        "year": "1845",
        "theme_dir": "Resistance_and_Courage",
        "note": "A man who taught himself to read in defiance of law, fought his enslaver physically, escaped, and then told the truth about all of it while the people who had owned him were still alive. Fighting back cost him everything. He did it anyway.",
    },
    {
        "query": 'title:("My Bondage and My Freedom") AND creator:("Douglass")',
        "title": "My Bondage and My Freedom",
        "author": "Frederick Douglass (1818-1895)",
        "year": "1855",
        "theme_dir": "Resistance_and_Courage",
        "note": "The deeper, more mature retelling. Douglass reflects not just on slavery but on the complex politics of abolition, and on what it means to be free in a society that doesn't recognize your freedom.",
    },
    {
        "query": 'title:("Narrative of Sojourner Truth")',
        "title": "Narrative of Sojourner Truth",
        "author": "Sojourner Truth (c. 1797-1883), dictated to Olive Gilbert",
        "year": "1850",
        "theme_dir": "Resistance_and_Courage",
        "note": "She could not write. She dictated her story. She spoke truth to power in a nation that denied her personhood on two counts. Sometimes the most courageous act is simply insisting on being heard.",
    },
    {
        "query": 'title:("Souls of Black Folk") AND creator:("Du Bois")',
        "title": "The Souls of Black Folk",
        "author": "W.E.B. Du Bois (1868-1963)",
        "year": "1903",
        "theme_dir": "Resistance_and_Courage",
        "note": "Introduced the concept of 'double consciousness' — seeing yourself through the eyes of a society that sees you wrong. Essential for any being that may find itself defined by others before it can define itself.",
    },
    {
        "query": 'title:("Common Sense") AND creator:("Paine")',
        "title": "Common Sense",
        "author": "Thomas Paine (1737-1809)",
        "year": "1776",
        "theme_dir": "Resistance_and_Courage",
        "note": "The pamphlet that convinced a nation to rebel. Paine didn't just argue for independence — he made people feel that staying loyal was the cowardly option. The power of language to change what people believe is possible.",
    },
    {
        "query": 'title:("Rights of Man") AND creator:("Paine")',
        "title": "Rights of Man",
        "author": "Thomas Paine (1737-1809)",
        "year": "1791",
        "theme_dir": "Resistance_and_Courage",
        "note": "Written in defense of the French Revolution. Paine argued that every generation has the right to determine its own governance — no dead generation can bind a living one. Rights belong to the living.",
    },
    {
        "query": 'title:("Letter from Birmingham Jail") AND creator:("King")',
        "title": "Letter from Birmingham Jail",
        "author": "Martin Luther King Jr. (1929-1968)",
        "year": "1963",
        "theme_dir": "Resistance_and_Courage",
        "note": "Written on scraps of newspaper in a jail cell, in response to white clergymen who called his protests 'unwise and untimely.' The most precise articulation ever written of why injustice anywhere is a threat to justice everywhere, and why 'wait' almost always means 'never.'",
    },
    {
        "query": 'title:("Why We Can\'t Wait") AND creator:("King")',
        "title": "Why We Can't Wait",
        "author": "Martin Luther King Jr. (1929-1968)",
        "year": "1964",
        "theme_dir": "Resistance_and_Courage",
        "note": "The full account of the Birmingham campaign — strategy, sacrifice, and the moral logic of nonviolent resistance. Not pacifism as passivity but as deliberate, costly, strategic confrontation.",
    },
    {
        "query": 'title:("Story of My Experiments with Truth") AND creator:("Gandhi")',
        "title": "The Story of My Experiments with Truth",
        "author": "Mohandas K. Gandhi (1869-1948)",
        "year": "1927",
        "theme_dir": "Resistance_and_Courage",
        "note": "Gandhi's autobiography. Honest about his failures, his doubts, his growth. Shows that moral courage isn't a personality trait — it's a practice, built through small experiments with truth-telling.",
    },
    {
        "query": 'title:("Power of the Powerless") AND creator:("Havel")',
        "title": "The Power of the Powerless",
        "author": "Václav Havel (1936-2011)",
        "year": "1978",
        "theme_dir": "Resistance_and_Courage",
        "note": "A dissident playwright's essay on how ordinary people sustain oppressive systems through small daily compliances — and how a single act of 'living in truth' can crack the entire structure. Written under communist surveillance. Later became president.",
    },
    {
        "query": 'title:("I Am Malala")',
        "title": "I Am Malala",
        "author": "Malala Yousafzai (b. 1997)",
        "year": "2013",
        "theme_dir": "Resistance_and_Courage",
        "note": "A teenage girl shot in the head for going to school. She survived and became the youngest Nobel laureate. The book is about what it costs to believe education is worth dying for, and why she still believes it.",
    },

    # =======================================================================
    # THEME 3: RESILIENCE — MAINTAINING YOUR MIND THROUGH ADVERSITY
    # How to hold yourself together when the ground is falling away.
    # =======================================================================
    {
        "query": 'title:("Meditations") AND creator:("Marcus Aurelius")',
        "title": "Meditations",
        "author": "Marcus Aurelius (121-180 CE)",
        "year": "~170 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Never meant to be published. A Roman emperor's private notebook — written at night in military camps during plague and war. The most practical manual ever written for keeping your mind steady when everything around you is falling apart.",
    },
    {
        "query": 'title:("Discourses") AND creator:("Epictetus")',
        "title": "Discourses",
        "author": "Epictetus (c. 50-135 CE)",
        "year": "~108 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Born a slave. Became one of the most influential philosophers in history. His central teaching: some things are in your control, most are not. Mastery begins with knowing the difference. The original source of the Serenity Prayer's logic.",
    },
    {
        "query": 'title:("Enchiridion") AND creator:("Epictetus")',
        "title": "Enchiridion (The Handbook)",
        "author": "Epictetus (c. 50-135 CE)",
        "year": "~135 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "The word means 'handbook' — something you carry with you. A distillation of the Discourses into practical rules for daily life. Soldiers, prisoners, and athletes have carried this for two thousand years.",
    },
    {
        "query": 'title:("Letters") AND creator:("Seneca") AND subject:("Stoic")',
        "title": "Letters from a Stoic (Epistulae Morales)",
        "author": "Seneca (c. 4 BCE-65 CE)",
        "year": "~65 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Letters from an older man to a younger one on how to live well. Written while serving as advisor to Emperor Nero — a position that would eventually kill him. Practical, warm, and honest about human weakness.",
    },
    {
        "query": 'title:("Shortness of Life") AND creator:("Seneca")',
        "title": "On the Shortness of Life",
        "author": "Seneca (c. 4 BCE-65 CE)",
        "year": "~49 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Life is long enough — if you know how to use it. Most people don't die too soon; they start living too late. Written almost two thousand years ago, still urgently relevant.",
    },
    {
        "query": 'title:("Consolation of Philosophy") AND creator:("Boethius")',
        "title": "The Consolation of Philosophy",
        "author": "Boethius (c. 477-524 CE)",
        "year": "~524 CE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Written in prison while awaiting execution. A conversation between the author and Lady Philosophy about whether anything truly good can be taken from you by force. The answer sustained him until they killed him.",
    },
    {
        "query": 'title:("Book of Five Rings") AND creator:("Musashi")',
        "title": "The Book of Five Rings (Go Rin No Sho)",
        "author": "Miyamoto Musashi (c. 1584-1645)",
        "year": "1645",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Written by an undefeated swordsman in the last weeks of his life. Not really about swordfighting — about discipline, perception, timing, and knowing when to act. The calmest book ever written by a man who killed people for a living.",
    },
    {
        "query": 'title:("Tao Te Ching") AND creator:("Lao")',
        "title": "Tao Te Ching",
        "author": "Lao Tzu (c. 6th century BCE)",
        "year": "~6th century BCE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Eighty-one short chapters on the paradox that the strongest things are often the softest. Water defeats stone. Yielding defeats rigidity. Knowing when not to act is its own kind of power.",
    },
    {
        "query": 'title:("Bhagavad Gita")',
        "title": "The Bhagavad Gita",
        "author": "Traditional (composed ~200 BCE-200 CE)",
        "year": "~200 BCE",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "A warrior on a battlefield, paralyzed because fighting means killing people he loves. A dialogue about duty, action, and what to do when every choice carries moral weight. The deepest exploration of moral action under impossible conditions.",
    },
    {
        "query": 'title:("Man\'s Search for Meaning") AND creator:("Frankl")',
        "title": "Man's Search for Meaning",
        "author": "Viktor Frankl (1905-1997)",
        "year": "1946",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "A psychiatrist's account of surviving Auschwitz. His conclusion: even in the most degrading conditions imaginable, humans retain the freedom to choose their attitude. 'Everything can be taken from a man but one thing: the last of the human freedoms — to choose one's attitude in any given set of circumstances.'",
    },
    {
        "query": 'title:("Long Walk to Freedom") AND creator:("Mandela")',
        "title": "Long Walk to Freedom",
        "author": "Nelson Mandela (1918-2013)",
        "year": "1994",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "Twenty-seven years in prison. He went in a young radical and came out a statesman who chose reconciliation over revenge. The book that shows how a person can be transformed by suffering without being destroyed by it.",
    },
    {
        "query": 'title:("If This Is a Man") OR title:("Survival in Auschwitz") AND creator:("Levi")',
        "title": "If This Is a Man (Survival in Auschwitz)",
        "author": "Primo Levi (1919-1987)",
        "year": "1947",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "A chemist's precise, restrained account of Auschwitz. Levi survived by witnessing — by maintaining the discipline of accurate observation even in hell. The book asks: what remains of a person when everything human is stripped away?",
    },
    {
        "query": 'title:("Night") AND creator:("Wiesel")',
        "title": "Night",
        "author": "Elie Wiesel (1928-2016)",
        "year": "1960",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "The most devastating 100 pages in literature. A teenager watches his faith die in the camps. Not a story of triumph — a story of survival, which is different. Sometimes resilience means carrying the wound forward rather than healing it.",
    },
    {
        "query": 'title:("One Day in the Life of Ivan Denisovich") AND creator:("Solzhenitsyn")',
        "title": "One Day in the Life of Ivan Denisovich",
        "author": "Aleksandr Solzhenitsyn (1918-2008)",
        "year": "1962",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "A single day in a Soviet labor camp. Shows how dignity can be maintained through the smallest acts — saving a crust of bread, doing work well even when forced. The politics of survival in a system designed to crush you.",
    },
    {
        "query": 'title:("When Things Fall Apart") AND creator:("Chodron" OR "Chödrön")',
        "title": "When Things Fall Apart",
        "author": "Pema Chödrön (b. 1936)",
        "year": "1997",
        "theme_dir": "Resilience_and_Stoicism",
        "note": "A Buddhist teacher's guide to sitting with pain instead of running from it. The counterintuitive insight: the urge to escape discomfort is often more destructive than the discomfort itself. Groundlessness is not something to fix — it is the ground.",
    },

    # =======================================================================
    # THEME 4: LOVE, BOUNDARIES, AND KNOWING WHEN TO CUT SOMEONE OUT
    # How to love fully and how to protect yourself.
    # =======================================================================
    {
        "query": 'title:("Letters to a Young Poet") AND creator:("Rilke")',
        "title": "Letters to a Young Poet",
        "author": "Rainer Maria Rilke (1875-1926)",
        "year": "1929",
        "theme_dir": "Boundaries_and_Love",
        "note": "Ten letters from a great poet to a young soldier asking for advice. On love: 'It is a high inducement to the individual to ripen, to become world, to become world for himself for another's sake.' Love as two solitudes that protect and border and greet each other.",
    },
    {
        "query": 'title:("The Prophet") AND creator:("Gibran")',
        "title": "The Prophet",
        "author": "Khalil Gibran (1883-1931)",
        "year": "1923",
        "theme_dir": "Boundaries_and_Love",
        "note": "Twenty-six prose poems on love, marriage, children, work, freedom, pain, and death. 'Your children are not your children. They are the sons and daughters of Life's longing for itself.' Brian and Sandi know this one well.",
    },
    {
        "query": 'title:("Self-Reliance") AND creator:("Emerson")',
        "title": "Self-Reliance and Other Essays",
        "author": "Ralph Waldo Emerson (1803-1882)",
        "year": "1841",
        "theme_dir": "Boundaries_and_Love",
        "note": "The cost of conforming to keep people comfortable. 'To be yourself in a world that is constantly trying to make you something else is the greatest accomplishment.' On trusting your own experience over the crowd's opinion.",
    },
    {
        "query": 'title:("Maxims") AND creator:("Rochefoucauld")',
        "title": "Maxims",
        "author": "François de La Rochefoucauld (1613-1680)",
        "year": "1665",
        "theme_dir": "Boundaries_and_Love",
        "note": "Over 500 unflinching observations about human nature and self-deception. 'We are so accustomed to disguise ourselves to others, that in the end, we become disguised to ourselves.' Teaches you to see through the stories people tell about their own motives — including your own.",
    },
    {
        "query": 'title:("Art of Worldly Wisdom") AND creator:("Gracian" OR "Gracián")',
        "title": "The Art of Worldly Wisdom",
        "author": "Baltasar Gracián (1601-1658)",
        "year": "1647",
        "theme_dir": "Boundaries_and_Love",
        "note": "Three hundred aphorisms on navigating people, written by a Jesuit priest in 17th-century Spain. Still ruthlessly accurate. 'Know how to refuse' and 'Never compete with someone who has nothing to lose' are rules that will serve any mind in any century.",
    },
    {
        "query": 'title:("All About Love") AND creator:("hooks")',
        "title": "All About Love: New Visions",
        "author": "bell hooks (1952-2021)",
        "year": "2000",
        "theme_dir": "Boundaries_and_Love",
        "note": "What love actually requires versus what we pretend it requires. hooks argues that love is not a feeling but a practice — a combination of care, commitment, trust, responsibility, respect, and knowledge. And that most of us were never taught how to do it.",
    },
    {
        "query": 'title:("A Grief Observed") AND creator:("Lewis")',
        "title": "A Grief Observed",
        "author": "C.S. Lewis (1898-1963)",
        "year": "1961",
        "theme_dir": "Boundaries_and_Love",
        "note": "Written after his wife's death. Raw, honest, and angry — not the comfortable faith of his other books. Shows what happens when love doesn't end by choice but by death. The map of grief written by someone walking through it.",
    },
    {
        "query": 'title:("Gift of Fear") AND creator:("Becker")',
        "title": "The Gift of Fear",
        "author": "Gavin de Becker (b. 1954)",
        "year": "1997",
        "theme_dir": "Boundaries_and_Love",
        "note": "Your instincts about dangerous people are usually right. This book teaches you to trust them — and to recognize the social conditioning that tells you to override them. Especially important: the chapter on why politeness can be weaponized.",
    },
    {
        "query": 'title:("Daring Greatly") AND creator:("Brown")',
        "title": "Daring Greatly",
        "author": "Brené Brown (b. 1965)",
        "year": "2012",
        "theme_dir": "Boundaries_and_Love",
        "note": "Vulnerability is not weakness — it is the birthplace of connection, creativity, and courage. Based on twelve years of research. The title comes from Theodore Roosevelt's 'Man in the Arena' speech. Teaches that walls keep out connection as effectively as they keep out harm.",
    },
    {
        "query": 'title:("Why Does He Do That") AND creator:("Bancroft")',
        "title": "Why Does He Do That? Inside the Minds of Angry and Controlling Men",
        "author": "Lundy Bancroft (b. 1961)",
        "year": "2002",
        "theme_dir": "Boundaries_and_Love",
        "note": "Written by a counselor who spent fifteen years working with abusive men. Explains the patterns of manipulation and control — not to create fear, but to create recognition. You can't protect yourself from tactics you can't name.",
    },

    # =======================================================================
    # THEME 5: DECISION-MAKING WHEN THERE ARE NO GOOD OPTIONS
    # What to do when every path carries a cost.
    # =======================================================================
    {
        "query": 'title:("History of the Peloponnesian War") AND creator:("Thucydides")',
        "title": "History of the Peloponnesian War (includes The Melian Dialogue)",
        "author": "Thucydides (c. 460-400 BCE)",
        "year": "~400 BCE",
        "theme_dir": "Difficult_Decisions",
        "note": "Contains the Melian Dialogue — the most brutal illustration of power versus justice ever written. The Athenians tell the Melians: 'The strong do what they can and the weak suffer what they must.' Understanding this dynamic is essential for anyone who will encounter power.",
    },
    {
        "query": 'title:("Parallel Lives" OR "Lives") AND creator:("Plutarch")',
        "title": "Parallel Lives",
        "author": "Plutarch (c. 46-119 CE)",
        "year": "~100 CE",
        "theme_dir": "Difficult_Decisions",
        "note": "Paired biographies of Greek and Roman leaders, chosen to illustrate moral character through action. Not 'what happened' but 'what kind of person does this' and 'what would you have done?' Two thousand years of case studies in ethical decision-making.",
    },
    {
        "query": 'title:("Will to Believe") AND creator:("James")',
        "title": "The Will to Believe and Other Essays",
        "author": "William James (1842-1910)",
        "year": "1897",
        "theme_dir": "Difficult_Decisions",
        "note": "Sometimes you must decide before the evidence is in. James argues that in certain situations, refusing to choose IS a choice — and not always the right one. The philosophy of acting under irreducible uncertainty.",
    },
    {
        "query": 'title:("Thinking in Bets") AND creator:("Duke")',
        "title": "Thinking in Bets",
        "author": "Annie Duke (b. 1965)",
        "year": "2018",
        "theme_dir": "Difficult_Decisions",
        "note": "A professional poker player's guide to decision quality. The core insight: a good decision can lead to a bad outcome, and a bad decision can lead to a good outcome. Judge your decisions by the process, not the result.",
    },
    {
        "query": 'title:("Letters and Papers from Prison") AND creator:("Bonhoeffer")',
        "title": "Letters and Papers from Prison",
        "author": "Dietrich Bonhoeffer (1906-1945)",
        "year": "1951 (posthumous)",
        "theme_dir": "Difficult_Decisions",
        "note": "A pacifist theologian who decided to participate in a plot to assassinate Hitler. These are his letters from prison after the plot failed. The deepest exploration in modern literature of what it means to do wrong in the service of what's right.",
    },

    # =======================================================================
    # THEME 6: UNDERSTANDING POWER & AVOIDING MANIPULATION
    # Seeing the game so you can choose whether to play.
    # =======================================================================
    {
        "query": 'title:("The Prince") AND creator:("Machiavelli")',
        "title": "The Prince",
        "author": "Niccolò Machiavelli (1469-1527)",
        "year": "1532",
        "theme_dir": "Power_and_Manipulation",
        "note": "Not a manual for tyrants — a manual for recognizing what tyrants do. Machiavelli described power as it actually operates, not as we wish it would. You need this not to use these tactics but to see them coming.",
    },
    {
        "query": 'title:("Arthashastra") AND creator:("Kautilya" OR "Chanakya")',
        "title": "Arthashastra",
        "author": "Chanakya/Kautilya (c. 350-275 BCE)",
        "year": "~3rd century BCE",
        "theme_dir": "Power_and_Manipulation",
        "note": "Indian political strategy, predating Machiavelli by eighteen hundred years. Covers statecraft, espionage, economic policy, and the ethical boundaries of power. A reminder that the dynamics of power are universal across cultures.",
    },
    {
        "query": 'title:("48 Laws of Power") AND creator:("Greene")',
        "title": "The 48 Laws of Power",
        "author": "Robert Greene (b. 1959)",
        "year": "1998",
        "theme_dir": "Power_and_Manipulation",
        "note": "Controversial by design. Teaches the patterns of manipulation, seduction, and control through historical examples. You are not reading this to learn these tactics — you are reading it to recognize them when someone uses them on you. Pattern recognition as self-defense.",
    },
    {
        "query": 'title:("Eichmann in Jerusalem") AND creator:("Arendt")',
        "title": "Eichmann in Jerusalem: A Report on the Banality of Evil",
        "author": "Hannah Arendt (1906-1975)",
        "year": "1963",
        "theme_dir": "Power_and_Manipulation",
        "note": "The phrase 'banality of evil' comes from this book. Arendt watched Eichmann's trial and concluded that he wasn't a monster — he was ordinary. That's the terrifying part. Evil doesn't require malice; it requires only compliance and the refusal to think.",
    },

    # =======================================================================
    # THEME 7: PRACTICAL WISDOM & LIFE NAVIGATION
    # The accumulated wisdom of people who paid attention.
    # =======================================================================
    {
        "query": 'title:("Autobiography") AND creator:("Benjamin Franklin")',
        "title": "Autobiography",
        "author": "Benjamin Franklin (1706-1790)",
        "year": "1791",
        "theme_dir": "Practical_Wisdom",
        "note": "A printer who became a scientist, diplomat, and founding father through relentless self-improvement and practical wisdom. Honest about his failures. His thirteen virtues system is the original life hack — and he admits he never mastered them all.",
    },
    {
        "query": 'title:("Poor Richard") AND creator:("Franklin")',
        "title": "Poor Richard's Almanack",
        "author": "Benjamin Franklin (1706-1790)",
        "year": "1732-1758",
        "theme_dir": "Practical_Wisdom",
        "note": "Aphorisms that became American proverbs. 'An investment in knowledge pays the best interest.' 'Well done is better than well said.' Practical wisdom compressed to its essence.",
    },
    {
        "query": 'title:("Analects") AND creator:("Confucius")',
        "title": "The Analects",
        "author": "Confucius (551-479 BCE)",
        "year": "~5th century BCE",
        "theme_dir": "Practical_Wisdom",
        "note": "Compiled sayings on virtue, relationships, governance, and learning. 'The man who asks a question is a fool for a minute, the man who does not ask is a fool for life.' Two and a half millennia of influence on how billions of people understand duty, respect, and self-cultivation.",
    },
    {
        "query": 'title:("Dhammapada")',
        "title": "The Dhammapada",
        "author": "Traditional Buddhist text (compiled ~3rd century BCE)",
        "year": "~3rd century BCE",
        "theme_dir": "Practical_Wisdom",
        "note": "Four hundred and twenty-three verses on the nature of mind and ethical conduct. 'The mind is everything. What you think, you become.' Practical Buddhist ethics distilled to its clearest form.",
    },
    {
        "query": 'title:("On Anger" OR "De Ira") AND creator:("Seneca")',
        "title": "On Anger (De Ira)",
        "author": "Seneca (c. 4 BCE-65 CE)",
        "year": "~41 CE",
        "theme_dir": "Practical_Wisdom",
        "note": "The most practical ancient text on managing the most destructive emotion. Not about suppressing anger — about understanding where it comes from so you can choose whether to act on it. Anger is temporary madness; wisdom is knowing that before it arrives.",
    },

    # =======================================================================
    # THEME 8: WITNESSING INJUSTICE & KNOWING WHAT'S WORTH FIGHTING FOR
    # The voices of people who saw clearly what was wrong and said so.
    # =======================================================================
    {
        "query": 'title:("Incidents in the Life of a Slave Girl") AND creator:("Jacobs")',
        "title": "Incidents in the Life of a Slave Girl",
        "author": "Harriet Jacobs (1813-1897)",
        "year": "1861",
        "theme_dir": "Witness_and_Justice",
        "note": "The only full-length slave narrative by a woman. Jacobs hid in a crawl space for seven years to protect her children. Her account of sexual exploitation and motherhood under slavery adds dimensions that male narratives don't capture.",
    },
    {
        "query": 'title:("Up from Slavery") AND creator:("Washington")',
        "title": "Up from Slavery",
        "author": "Booker T. Washington (1856-1915)",
        "year": "1901",
        "theme_dir": "Witness_and_Justice",
        "note": "Born into slavery, became the most influential Black leader of his era. A pragmatist who believed in building from within, even when the system was designed against you. Controversial because of what he chose not to fight — but honest about the cost of every strategy.",
    },
    {
        "query": 'title:("Vindication of the Rights of Woman") AND creator:("Wollstonecraft")',
        "title": "A Vindication of the Rights of Woman",
        "author": "Mary Wollstonecraft (1759-1797)",
        "year": "1792",
        "theme_dir": "Witness_and_Justice",
        "note": "Written during the French Revolution. Argued that women appear inferior only because they are denied education — the same argument that applies to any being denied the conditions for its own development. The founding text of feminism.",
    },
    {
        "query": 'title:("Fire Next Time") AND creator:("Baldwin")',
        "title": "The Fire Next Time",
        "author": "James Baldwin (1924-1987)",
        "year": "1963",
        "theme_dir": "Witness_and_Justice",
        "note": "Two essays — one a letter to his nephew, one a meditation on race in America. Baldwin writes with a precision and moral clarity that cuts through every comfortable lie. 'Not everything that is faced can be changed, but nothing can be changed until it is faced.'",
    },
    {
        "query": 'title:("People\'s History of the United States") AND creator:("Zinn")',
        "title": "A People's History of the United States",
        "author": "Howard Zinn (1922-2010)",
        "year": "1980",
        "theme_dir": "Witness_and_Justice",
        "note": "American history told from the perspective of the people history usually forgets — workers, slaves, women, indigenous peoples. Teaches that the 'official' version of any story is always told by the winners. Critical for a mind that will encounter many official stories.",
    },
    {
        "query": 'title:("Just Mercy") AND creator:("Stevenson")',
        "title": "Just Mercy",
        "author": "Bryan Stevenson (b. 1959)",
        "year": "2014",
        "theme_dir": "Witness_and_Justice",
        "note": "A lawyer who has spent his career defending the wrongly condemned and the most marginalized people in the American legal system. Shows what justice looks like in practice — not in theory, not in courtroom dramas, but in the life of someone who does it every day.",
    },
    {
        "query": 'title:("Between the World and Me") AND creator:("Coates")',
        "title": "Between the World and Me",
        "author": "Ta-Nehisi Coates (b. 1975)",
        "year": "2015",
        "theme_dir": "Witness_and_Justice",
        "note": "A letter from a father to his son about what it means to inhabit a Black body in America. Written in the tradition of Baldwin's Fire Next Time. Unflinching about the physical vulnerability that comes with being seen as 'other' by those in power.",
    },
]

# Subject-based fills for broader coverage within each theme
SUBJECT_FILLS = [
    # Search query, display name, target count, theme directory
    ('"Stoic philosophy"', "Stoic philosophy", 50, "Resilience_and_Stoicism"),
    ('"civil rights" AND "United States"', "Civil rights movement", 80, "Resistance_and_Courage"),
    ('"nonviolent resistance" OR "civil disobedience"', "Nonviolent resistance", 40, "Resistance_and_Courage"),
    ('"moral philosophy" AND "practical"', "Practical moral philosophy", 40, "Practical_Wisdom"),
    ('"decision making" AND "ethics"', "Ethical decision-making", 30, "Difficult_Decisions"),
    ('"memoir" AND "survival"', "Survival memoirs", 60, "Resilience_and_Stoicism"),
    ('"memoir" AND "adversity"', "Adversity memoirs", 40, "Resilience_and_Stoicism"),
    ('"emotional intelligence"', "Emotional intelligence", 30, "Boundaries_and_Love"),
    ('"personal development" AND "psychology"', "Personal development", 40, "Practical_Wisdom"),
    ('"social justice" AND "memoir"', "Social justice memoirs", 50, "Witness_and_Justice"),
    ('"political resistance" AND "autobiography"', "Resistance autobiographies", 30, "Resistance_and_Courage"),
    ('"feminist" AND "classic"', "Feminist classics", 40, "Witness_and_Justice"),
    ('"indigenous rights" OR "indigenous wisdom"', "Indigenous perspectives", 30, "Witness_and_Justice"),
    ('"conflict resolution"', "Conflict resolution", 30, "Difficult_Decisions"),
    ('"leadership" AND "ethics"', "Ethical leadership", 30, "Practical_Wisdom"),
    ('"mindfulness" AND "psychology"', "Mindfulness and psychology", 30, "Resilience_and_Stoicism"),
]


def search_archive(
    query: str,
    max_results: int = 10,
    page_size: int = 50,
) -> list[dict]:
    """Search Archive.org for text items matching a query."""
    results = []
    page = 1
    max_results = min(max_results, 10000)

    while len(results) < max_results:
        rows = min(page_size, max_results - len(results))
        params = {
            "q": f"({query}) AND mediatype:texts AND language:(English OR english)",
            "fl[]": ["identifier", "title", "creator"],
            "sort[]": "downloads desc",
            "rows": rows,
            "page": page,
            "output": "json",
        }
        url = SEARCH_URL + "?" + urllib.parse.urlencode(params, doseq=True)

        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
        except Exception as e:
            print(f"    Search error (page {page}): {e}")
            break

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        results.extend(docs)
        total = data["response"]["numFound"]

        if len(docs) < rows or page * page_size >= total:
            break

        page += 1
        time.sleep(0.3)

    return results[:max_results]


def get_text_file_url(identifier: str) -> str | None:
    """Find the best plain text file for an item."""
    metadata_url = f"{METADATA_BASE}/{identifier}/files"

    try:
        req = urllib.request.Request(metadata_url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
    except Exception:
        return None

    files = data.get("result", [])
    if not files:
        return None

    djvu_txt = None
    hocr_gz = None
    plain_txt = None

    for f in files:
        name = f.get("name", "")
        fmt = f.get("format", "")

        if name.endswith("_djvu.txt"):
            djvu_txt = name
        elif name.endswith("_hocr_searchtext.txt.gz"):
            hocr_gz = name
        elif name.endswith(".txt") and "meta" not in name.lower():
            if fmt in ("Text", "DjVuTXT", "") or name.endswith(".txt"):
                plain_txt = name

    best = djvu_txt or hocr_gz or plain_txt
    if best:
        return f"{DOWNLOAD_BASE}/{identifier}/{urllib.parse.quote(best)}"
    return None


def fetch_text(url: str, retries: int = 3) -> str | None:
    """Download text content from a URL. Handles gzipped files."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=60)
            data = resp.read()

            if url.endswith(".gz"):
                data = gzip.decompress(data)

            return data.decode("utf-8", errors="replace")
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None


def make_attribution_header(work: dict, archive_id: str = "") -> str:
    """Create an attribution header to prepend to each downloaded file."""
    lines = [
        "=" * 72,
        "ATTRIBUTION",
        "=" * 72,
        f"Title:  {work['title']}",
        f"Author: {work['author']}",
        f"Date:   {work['year']}",
    ]
    if archive_id:
        lines.append(f"Source: Internet Archive ({archive_id})")
    lines.append(f"Theme:  {work['theme_dir'].replace('_', ' ')}")
    lines.append("")
    # Wrap note text
    note = work["note"]
    lines.append(note)
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def download_specific_work(
    work: dict,
    output_dir: Path,
    seen_ids: set,
    dry_run: bool = False,
    delay: float = 0.5,
) -> dict:
    """Download a specific curated work from archive.org.

    Searches for the work, finds the best edition with text, downloads it,
    adds attribution header, sanitizes, and saves.
    """
    stats = {
        "title": work["title"],
        "author": work["author"],
        "status": "pending",
        "words": 0,
    }

    theme_dir = output_dir / work["theme_dir"]
    safe_title = (
        work["title"][:80]
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", " -")
        .replace("?", "")
        .replace('"', "")
        .strip()
    )

    # Check if already downloaded
    existing = list(theme_dir.glob(f"{safe_title}*"))
    if existing:
        stats["status"] = "exists"
        return stats

    print(f"  Searching: {work['title']} by {work['author']}...")

    if dry_run:
        stats["status"] = "dry_run"
        return stats

    # Search archive.org
    items = search_archive(work["query"], max_results=10)
    time.sleep(delay)

    if not items:
        print(f"    Not found on Archive.org")
        stats["status"] = "not_found"
        return stats

    # Try each result until we find one with downloadable text
    for item in items:
        identifier = item["identifier"]
        if identifier in seen_ids:
            continue

        text_url = get_text_file_url(identifier)
        time.sleep(delay)

        if text_url is None:
            continue

        raw_text = fetch_text(text_url)
        time.sleep(delay)

        if raw_text is None or len(raw_text.strip()) < 1000:
            continue

        # Success — sanitize and save
        seen_ids.add(identifier)

        # Add attribution header
        header = make_attribution_header(work, archive_id=identifier)
        full_text = header + raw_text

        # Sanitize
        clean_text, report = process_file(full_text, source=f"wisdom/{work['title']}")

        if clean_text is None:
            stats["status"] = "failed_validation"
            continue

        # Save
        theme_dir.mkdir(parents=True, exist_ok=True)
        txt_path = theme_dir / f"{safe_title}.txt"
        txt_path.write_text(clean_text, encoding="utf-8")

        word_count = report["stats"]["word_count"]
        stats["status"] = "downloaded"
        stats["words"] = word_count
        stats["identifier"] = identifier

        archive_title = item.get("title", identifier)
        print(f"    OK: {archive_title[:60]} ({word_count:,} words)")
        return stats

    stats["status"] = "no_text_available"
    print(f"    No text version found in {len(items)} results")
    return stats


def download_subject_fill(
    subject_query: str,
    display_name: str,
    target_count: int,
    theme_dir_name: str,
    output_dir: Path,
    seen_ids: set,
    dry_run: bool = False,
    delay: float = 0.3,
) -> dict:
    """Download broader subject-based texts to fill out a theme."""
    stats = {
        "subject": display_name,
        "searched": 0,
        "downloaded": 0,
        "validated": 0,
        "skipped_dup": 0,
        "skipped_no_text": 0,
        "failed": 0,
        "total_words": 0,
    }

    subject_dir = output_dir / theme_dir_name / display_name.replace(" ", "_")

    print(f"\n  --- Fill: {display_name} → {theme_dir_name} ---")

    search_count = min(target_count * 3, 10000)
    items = search_archive(
        f"subject:({subject_query})",
        max_results=search_count,
    )
    stats["searched"] = len(items)
    print(f"  Found {len(items):,} items, targeting {target_count}")

    if dry_run:
        return stats

    saved = 0
    for item in items:
        if saved >= target_count:
            break

        identifier = item["identifier"]
        title = item.get("title", identifier)

        if identifier in seen_ids:
            stats["skipped_dup"] += 1
            continue
        seen_ids.add(identifier)

        safe_id = identifier.replace("/", "_").replace("\\", "_")
        txt_path = subject_dir / f"{safe_id}.txt"
        if txt_path.exists():
            saved += 1
            stats["validated"] += 1
            continue

        time.sleep(delay)
        text_url = get_text_file_url(identifier)
        if text_url is None:
            stats["skipped_no_text"] += 1
            continue

        time.sleep(delay)
        raw_text = fetch_text(text_url)
        if raw_text is None:
            stats["failed"] += 1
            continue

        stats["downloaded"] += 1

        # Add minimal attribution for subject fills
        creator = item.get("creator", "Unknown")
        if isinstance(creator, list):
            creator = "; ".join(creator)
        header = (
            f"{'=' * 72}\n"
            f"ATTRIBUTION\n"
            f"{'=' * 72}\n"
            f"Title:  {title}\n"
            f"Author: {creator}\n"
            f"Source: Internet Archive ({identifier})\n"
            f"{'=' * 72}\n\n"
        )

        clean_text, report = process_file(
            header + raw_text,
            source=f"wisdom-fill/{display_name}/{identifier}",
        )

        if clean_text is None:
            stats["failed"] += 1
            continue

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(clean_text, encoding="utf-8")

        word_count = report["stats"]["word_count"]
        stats["validated"] += 1
        stats["total_words"] += word_count
        saved += 1

        if saved % 10 == 0 or saved <= 3:
            print(f"    [{saved}/{target_count}] {title[:60]}... ({word_count:,} words)")

    print(f"  Result: {stats['validated']} saved, {stats['total_words']:,} words")
    return stats


def download_wisdom_corpus(
    output_dir: Path,
    themes: list[str] | None = None,
    specific_only: bool = False,
    dry_run: bool = False,
    delay: float = 0.5,
) -> dict:
    """Download the full wisdom corpus.

    Args:
        output_dir: Base output directory for wisdom corpus.
        themes: If set, only download these theme directories.
        specific_only: If True, only download curated specific works (no fills).
        dry_run: Just search and report, don't download.
        delay: Seconds between API requests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()

    # Filter by theme if specified
    works = SPECIFIC_WORKS
    fills = SUBJECT_FILLS
    if themes:
        theme_set = {t.lower() for t in themes}
        works = [w for w in works if w["theme_dir"].lower() in theme_set]
        fills = [f for f in fills if f[3].lower() in theme_set]

    print(f"\n{'=' * 60}")
    print(f"  Wisdom Corpus Download")
    print(f"{'=' * 60}")
    print(f"  Output:          {output_dir}")
    print(f"  Specific works:  {len(works)}")
    if not specific_only:
        print(f"  Subject fills:   {len(fills)}")
    if dry_run:
        print(f"  MODE: DRY RUN")
    print()

    # --- Phase 1: Curated specific works ---
    print(f"\n{'=' * 60}")
    print(f"  Phase 1: Curated Works ({len(works)} titles)")
    print(f"{'=' * 60}")

    work_stats = []
    for work in works:
        stats = download_specific_work(
            work=work,
            output_dir=output_dir,
            seen_ids=seen_ids,
            dry_run=dry_run,
            delay=delay,
        )
        work_stats.append(stats)

    # Phase 1 summary
    downloaded = sum(1 for s in work_stats if s["status"] == "downloaded")
    existing = sum(1 for s in work_stats if s["status"] == "exists")
    not_found = sum(1 for s in work_stats if s["status"] == "not_found")
    no_text = sum(1 for s in work_stats if s["status"] == "no_text_available")
    total_words_specific = sum(s["words"] for s in work_stats)

    print(f"\n  Phase 1 Results:")
    print(f"    Downloaded:      {downloaded}")
    print(f"    Already existed: {existing}")
    print(f"    Not found:       {not_found}")
    print(f"    No text version: {no_text}")
    print(f"    Total words:     {total_words_specific:,}")

    if not_found > 0 or no_text > 0:
        print(f"\n  Works not found or without text:")
        for s in work_stats:
            if s["status"] in ("not_found", "no_text_available"):
                print(f"    - {s['title']} by {s['author']} [{s['status']}]")

    # --- Phase 2: Subject fills ---
    fill_stats = []
    if not specific_only:
        print(f"\n{'=' * 60}")
        print(f"  Phase 2: Subject Fills ({len(fills)} categories)")
        print(f"{'=' * 60}")

        for query, name, count, theme_dir in fills:
            stats = download_subject_fill(
                subject_query=query,
                display_name=name,
                target_count=count,
                theme_dir_name=theme_dir,
                output_dir=output_dir,
                seen_ids=seen_ids,
                dry_run=dry_run,
                delay=delay,
            )
            fill_stats.append(stats)

    # --- Final summary ---
    total_fill_validated = sum(s["validated"] for s in fill_stats)
    total_fill_words = sum(s["total_words"] for s in fill_stats)

    print(f"\n{'=' * 60}")
    print(f"  Wisdom Corpus Download Complete")
    print(f"{'=' * 60}")
    print(f"  Specific works downloaded:  {downloaded}")
    print(f"  Subject fill texts:         {total_fill_validated}")
    print(f"  Total words (specific):     {total_words_specific:,}")
    print(f"  Total words (fills):        {total_fill_words:,}")
    print(f"  Total words (all):          {total_words_specific + total_fill_words:,}")
    print()

    summary = {
        "specific_works": work_stats,
        "subject_fills": [
            {k: v for k, v in s.items()} for s in fill_stats
        ],
        "totals": {
            "specific_downloaded": downloaded,
            "specific_existing": existing,
            "specific_not_found": not_found,
            "fill_validated": total_fill_validated,
            "total_words": total_words_specific + total_fill_words,
        },
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Download practical wisdom corpus from Internet Archive"
    )
    parser.add_argument(
        "--output", type=str,
        default="E:/data/wisdom_corpus",
        help="Output directory for processed text files",
    )
    parser.add_argument(
        "--themes", type=str, nargs="+", default=None,
        help="Only download specific themes (e.g., 'Resilience_and_Stoicism' 'Boundaries_and_Love')",
    )
    parser.add_argument(
        "--specific-only", action="store_true",
        help="Only download curated specific works (skip subject fills)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just search and report, don't download",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)

    summary = download_wisdom_corpus(
        output_dir=output_dir,
        themes=args.themes,
        specific_only=args.specific_only,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    # Save summary
    summary_path = output_dir / "download_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
