import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


DESCRIPTION_WEIGHT = 0.30
GOALS_WEIGHT = 0.60
COURSE_SIZE_WEIGHT = 0.10
US_CREDIT_TO_ECTS = 2.0
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_N = 5
SIZE_TO_ECTS = {
    "ECTS": 1.0,
    "US semester credit hours": US_CREDIT_TO_ECTS,
}


def read_courses(dataset_path):
    courses = []

    with dataset_path.open(encoding="utf-8", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        for row in reader:
            course_id = row["course_id"].strip()
            course_size = float(row["course_size"])
            course_size_unit = row["course_size_unit"].strip()
            courses.append({
                "course_id": course_id,
                "university": row["university"].strip(),
                "title": row["title"].strip(),
                "description": row["description"].strip(),
                "learning_goals": row["learning_goals"].strip(),
                "course_size": course_size,
                "course_size_unit": course_size_unit,
                "size_ects_equivalent": (
                    course_size * SIZE_TO_ECTS[course_size_unit]
                ),
            })

    return courses


def compare_courses(courses, selected_course_id, model):
    selected_index = next(
        index for index, course in enumerate(courses)
        if course["course_id"].casefold() == selected_course_id.casefold()
    )

    description_vectors = model.encode(
        [course["description"] for course in courses],
        normalize_embeddings=True,
    )
    goal_vectors = model.encode(
        [course["learning_goals"] for course in courses],
        normalize_embeddings=True,
    )
    selected = courses[selected_index]
    results = []

    for index, course in enumerate(courses):
        if index == selected_index:
            continue
        description_score = max(
            0.0,
            float(util.cos_sim(
                description_vectors[selected_index], description_vectors[index]
            ).item()),
        )
        goals_score = max(
            0.0,
            float(util.cos_sim(
                goal_vectors[selected_index], goal_vectors[index]
            ).item()),
        )
        course_size_score = 1 - abs(
            selected["size_ects_equivalent"] - course["size_ects_equivalent"]
        ) / max(
            selected["size_ects_equivalent"], course["size_ects_equivalent"]
        )
        overall_score = (
            DESCRIPTION_WEIGHT * description_score
            + GOALS_WEIGHT * goals_score
            + COURSE_SIZE_WEIGHT * course_size_score
        )
        results.append({
            "course": course,
            "overall": overall_score,
            "description": description_score,
            "goals": goals_score,
            "course_size": course_size_score,
        })

    ranked_results = sorted(
        results, key=lambda result: result["overall"], reverse=True
    )
    return selected, ranked_results[:TOP_N]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rank courses with sentence embeddings for text and a numeric "
            "course-size similarity."
        )
    )
    parser.add_argument("course_id", help="ID of the course to compare")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("courses.csv"),
        help="Path to a course CSV (default: courses.csv beside this script)",
    )
    args = parser.parse_args()

    load_dotenv(Path(__file__).with_name(".env"))

    courses = read_courses(args.dataset)
    model = SentenceTransformer(MODEL_NAME, token=os.getenv("HF_TOKEN") or None)
    selected, results = compare_courses(courses, args.course_id, model)

    print(
        f"Comparisons for {selected['course_id']} - {selected['title']} "
        f"({selected['university']}, {selected['course_size']:g} "
        f"{selected['course_size_unit']})"
    )
    print(f"Text model: {MODEL_NAME}")
    print(f"Top {TOP_N} most similar courses:")
    print(
        "Overall weights: learning goals 60%, descriptions 30%, course size 10%."
    )
    print("Scores are weighted similarities from 0% (low) to 100% (high).")
    print(
        "Course-size scores approximate 1 US semester credit hour as 2 ECTS."
    )
    for result in results:
        course = result["course"]
        print(
            f"\n{course['course_id']} - {course['title']} "
            f"({course['university']})"
            f"\n  Overall:      {result['overall']:.1%}"
            f"\n  Description:  {result['description']:.1%}"
            f"\n  Learning goals: {result['goals']:.1%}"
            f"\n  Course size:  {result['course_size']:.1%} "
            f"({course['course_size']:g} {course['course_size_unit']})"
        )


if __name__ == "__main__":
    sys.exit(main())
