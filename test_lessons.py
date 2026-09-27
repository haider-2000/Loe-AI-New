import asyncio
import os
import tempfile

from database import create_lesson, enroll_student, init_db, lesson_roster, list_lessons


async def main() -> None:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        await init_db(path)
        lesson_id = await create_lesson(path, "رياضيات", "2026-10-01", "10:00")
        for index in range(50):
            ok, message = await enroll_student(path, lesson_id, f"S{index + 1:02d}")
            assert ok, (index, message)
        ok, message = await enroll_student(path, lesson_id, "S51")
        assert not ok and "مكتملة" in message, message
        ok, message = await enroll_student(path, lesson_id, "S01")
        assert not ok and "مسبقًا" in message, message
        rows = await list_lessons(path)
        assert rows[0]["enrolled"] == 50 and rows[0]["capacity"] == 50
        roster = await lesson_roster(path, lesson_id)
        assert roster is not None and len(roster["students"]) == 50
        print("PASS: 50 students enrolled; full and duplicate checks passed")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    asyncio.run(main())
