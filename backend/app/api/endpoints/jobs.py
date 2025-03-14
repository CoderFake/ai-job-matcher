"""
API endpoints cho tìm kiếm và quản lý công việc
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.models.cv import CVData
from app.models.job import JobData, JobMatch
from app.services.job_search.web_search import WebSearcher
from app.services.job_search.crawler import WebCrawler
from app.services.job_search.matcher import JobMatcher

logger = get_logger("api")
router = APIRouter()

# Khởi tạo các đối tượng dịch vụ
web_searcher = WebSearcher()
job_matcher = JobMatcher()


class JobSearchRequest(BaseModel):
    """Yêu cầu tìm kiếm việc làm"""
    cv_id: Optional[str] = Field(None, description="ID của CV để tìm kiếm việc làm phù hợp")
    query: Dict[str, Any] = Field(..., description="Truy vấn tìm kiếm")
    max_results: int = Field(20, description="Số lượng kết quả tối đa")


class JobMatchRequest(BaseModel):
    """Yêu cầu phân tích độ phù hợp của CV với công việc"""
    cv_id: str = Field(..., description="ID của CV")
    job_ids: List[str] = Field(..., description="Danh sách ID công việc")


class JobDetailRequest(BaseModel):
    """Yêu cầu thông tin chi tiết về công việc"""
    job_id: str = Field(..., description="ID của công việc")


@router.post("/search", response_model=Dict[str, Any], summary="Tìm kiếm việc làm")
async def search_jobs(
        request: JobSearchRequest,
        background_tasks: BackgroundTasks
):
    """
    Tìm kiếm việc làm dựa trên truy vấn hoặc CV.

    - **cv_id**: ID của CV để tìm kiếm việc làm phù hợp (tùy chọn)
    - **query**: Truy vấn tìm kiếm
    - **max_results**: Số lượng kết quả tối đa

    Trả về danh sách các công việc phù hợp.
    """
    try:
        start_time = time.time()

        # Nếu có CV, lấy dữ liệu CV
        cv_data = None
        if request.cv_id:
            if not hasattr(router, "analyzed_cvs") or request.cv_id not in router.analyzed_cvs:
                raise HTTPException(
                    status_code=404,
                    detail=f"Không tìm thấy dữ liệu phân tích CV với ID: {request.cv_id}"
                )

            cv_data = router.analyzed_cvs[request.cv_id]

            # Tạo truy vấn từ CV
            if not request.query:
                request.query = cv_data.to_job_search_query()

        # Thực hiện tìm kiếm
        jobs = web_searcher.search(request.query, request.max_results)

        # Lưu kết quả tìm kiếm
        search_id = f"search_{int(time.time())}"
        if not hasattr(router, "search_results"):
            router.search_results = {}

        router.search_results[search_id] = jobs

        # Phân tích độ phù hợp nếu có CV
        matches = []
        if cv_data and jobs:
            matches = job_matcher.match_multiple(cv_data, jobs)

            # Lưu kết quả phân tích
            if not hasattr(router, "job_matches"):
                router.job_matches = {}

            router.job_matches[search_id] = matches

        # Bắt đầu crawl chi tiết công việc trong nền nếu có kết quả
        if jobs:
            background_tasks.add_task(crawl_job_details, jobs)

        # Phân tích yêu cầu công việc nếu có CV
        job_requirements_analysis = None
        if cv_data and jobs:
            job_requirements_analysis = job_matcher.analyze_job_requirements(cv_data, jobs)

        # Tạo đề xuất kỹ năng nếu có CV
        skill_recommendations = None
        if cv_data and jobs:
            skill_recommendations = job_matcher.get_skill_recommendations(cv_data, jobs)

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        return {
            "search_id": search_id,
            "query": request.query,
            "total_results": len(jobs),
            "jobs": jobs,
            "matches": matches if matches else None,
            "job_requirements_analysis": job_requirements_analysis,
            "skill_recommendations": skill_recommendations,
            "processing_time": processing_time
        }

    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm việc làm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tìm kiếm việc làm: {str(e)}"
        )


async def crawl_job_details(jobs: List[JobData]):
    """
    Hàm crawl chi tiết công việc trong nền
    """
    try:
        # Khởi tạo crawler
        crawler = WebCrawler()

        # Lấy danh sách URL cần crawl
        urls = [job.source_url for job in jobs if job.source_url]

        # Thực hiện crawl
        if urls:
            job_details = await crawler.crawl_jobs(urls)

            # Cập nhật thông tin công việc
            if job_details:
                # Lưu thông tin chi tiết
                if not hasattr(router, "job_details"):
                    router.job_details = {}

                for job in job_details:
                    # Lưu theo URL
                    if job.source_url:
                        router.job_details[job.source_url] = job

                        # Cập nhật thông tin công việc trong kết quả tìm kiếm
                        if hasattr(router, "search_results"):
                            for search_id, search_jobs in router.search_results.items():
                                for i, search_job in enumerate(search_jobs):
                                    if search_job.source_url == job.source_url:
                                        # Cập nhật thông tin
                                        search_jobs[i] = job
    except Exception as e:
        logger.error(f"Lỗi khi crawl chi tiết công việc: {str(e)}")


@router.get("/details/{job_id}", response_model=Dict[str, Any], summary="Lấy thông tin chi tiết về công việc")
async def get_job_details(
        job_id: str
):
    """
    Lấy thông tin chi tiết về công việc.

    - **job_id**: ID của công việc

    Trả về thông tin chi tiết về công việc.
    """
    try:
        # Tìm kiếm công việc trong kết quả tìm kiếm
        job = None

        if hasattr(router, "search_results"):
            for search_id, jobs in router.search_results.items():
                for search_job in jobs:
                    if search_job.id == job_id:
                        job = search_job
                        break
                if job:
                    break

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy công việc với ID: {job_id}"
            )

        # Kiểm tra xem có thông tin chi tiết không
        detailed_job = None
        if hasattr(router, "job_details") and job.source_url in router.job_details:
            detailed_job = router.job_details[job.source_url]

        # Thu thập thêm thông tin về công ty
        company_info = None
        if job.company and job.company.name:
            # Khởi tạo crawler
            crawler = WebCrawler()

            # Lấy thông tin công ty
            company_info = await crawler.get_company_info(job.company.name)

        # Thu thập thông tin về xu hướng công việc
        job_trends = None
        if job.title:
            # Khởi tạo crawler
            crawler = WebCrawler()

            # Lấy thông tin xu hướng
            job_trends = await crawler.crawl_job_trends(job.title)

        # Thu thập thông tin về mức lương
        salary_data = None
        if job.title:
            # Khởi tạo crawler
            crawler = WebCrawler()

            # Lấy thông tin mức lương
            salary_data = await crawler.crawl_salary_data(job.title)

        # Thu thập thông tin về địa điểm
        location_info = None
        if job.location and job.location.city:
            # Khởi tạo crawler
            crawler = WebCrawler()

            # Lấy thông tin địa điểm
            location_info = await crawler.get_location_info(job.location.city)

        return {
            "job_id": job_id,
            "job": detailed_job or job,
            "company_info": company_info,
            "job_trends": job_trends,
            "salary_data": salary_data,
            "location_info": location_info
        }

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin chi tiết về công việc: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin chi tiết về công việc: {str(e)}"
        )


@router.post("/match", response_model=List[JobMatch], summary="Phân tích độ phù hợp của CV với công việc")
async def match_cv_with_jobs(
        request: JobMatchRequest
):
    """
    Phân tích độ phù hợp của CV với danh sách công việc.

    - **cv_id**: ID của CV
    - **job_ids**: Danh sách ID công việc

    Trả về danh sách kết quả phân tích độ phù hợp.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "analyzed_cvs") or request.cv_id not in router.analyzed_cvs:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dữ liệu phân tích CV với ID: {request.cv_id}"
            )

        # Lấy dữ liệu CV
        cv_data = router.analyzed_cvs[request.cv_id]

        # Tìm các công việc
        jobs = []
        if hasattr(router, "search_results"):
            for search_id, search_jobs in router.search_results.items():
                for job in search_jobs:
                    if job.id in request.job_ids:
                        jobs.append(job)

        if not jobs:
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy công việc nào với các ID đã cung cấp"
            )

        # Phân tích độ phù hợp
        matches = job_matcher.match_multiple(cv_data, jobs)

        return matches

    except Exception as e:
        logger.error(f"Lỗi khi phân tích độ phù hợp: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích độ phù hợp: {str(e)}"
        )


@router.get("/search/{search_id}", response_model=Dict[str, Any], summary="Lấy kết quả tìm kiếm đã lưu")
async def get_search_results(
        search_id: str
):
    """
    Lấy kết quả tìm kiếm đã lưu.

    - **search_id**: ID của kết quả tìm kiếm

    Trả về kết quả tìm kiếm đã lưu.
    """
    try:
        # Kiểm tra xem kết quả tìm kiếm có tồn tại không
        if not hasattr(router, "search_results") or search_id not in router.search_results:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {search_id}"
            )

        # Lấy kết quả tìm kiếm
        jobs = router.search_results[search_id]

        # Lấy kết quả phân tích độ phù hợp nếu có
        matches = None
        if hasattr(router, "job_matches") and search_id in router.job_matches:
            matches = router.job_matches[search_id]

        return {
            "search_id": search_id,
            "total_results": len(jobs),
            "jobs": jobs,
            "matches": matches
        }

    except Exception as e:
        logger.error(f"Lỗi khi lấy kết quả tìm kiếm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy kết quả tìm kiếm: {str(e)}"
        )


@router.get("/similar/{job_id}", response_model=List[JobData], summary="Tìm các công việc tương tự")
async def find_similar_jobs(
        job_id: str,
        limit: int = Query(5, description="Số lượng kết quả tối đa")
):
    """
    Tìm các công việc tương tự với một công việc cụ thể.

    - **job_id**: ID của công việc
    - **limit**: Số lượng kết quả tối đa (mặc định là 5)

    Trả về danh sách các công việc tương tự.
    """
    try:
        # Tìm kiếm công việc trong kết quả tìm kiếm
        job = None
        all_jobs = []

        if hasattr(router, "search_results"):
            for search_id, jobs in router.search_results.items():
                all_jobs.extend(jobs)
                for search_job in jobs:
                    if search_job.id == job_id:
                        job = search_job
                        break
                if job:
                    break

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy công việc với ID: {job_id}"
            )

        # Sử dụng WebSearcher để tìm các công việc tương tự
        similar_jobs = web_searcher.find_similar_jobs(job, all_jobs, limit)

        return similar_jobs

    except Exception as e:
        logger.error(f"Lỗi khi tìm công việc tương tự: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tìm công việc tương tự: {str(e)}"
        )


@router.get("/trends/{job_title}", response_model=Dict[str, Any], summary="Lấy thông tin xu hướng công việc")
async def get_job_trends(
        job_title: str
):
    """
    Lấy thông tin về xu hướng công việc.

    - **job_title**: Tên vị trí công việc

    Trả về thông tin về xu hướng công việc.
    """
    try:
        # Khởi tạo crawler
        crawler = WebCrawler()

        # Lấy thông tin xu hướng
        job_trends = await crawler.crawl_job_trends(job_title)

        # Lấy các vị trí tương tự
        similar_jobs = await crawler.crawl_similar_jobs(job_title)

        # Lấy các kỹ năng yêu cầu
        required_skills = await crawler.crawl_required_skills(job_title)

        # Lấy thông tin mức lương
        salary_data = await crawler.crawl_salary_data(job_title)

        return {
            "job_title": job_title,
            "trends": job_trends,
            "similar_jobs": similar_jobs,
            "required_skills": required_skills,
            "salary_data": salary_data
        }

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin xu hướng công việc: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin xu hướng công việc: {str(e)}"
        )


@router.get("/company/{company_name}", response_model=Dict[str, Any], summary="Lấy thông tin công ty")
async def get_company_info(
        company_name: str
):
    """
    Lấy thông tin về công ty.

    - **company_name**: Tên công ty

    Trả về thông tin về công ty.
    """
    try:
        # Khởi tạo crawler
        crawler = WebCrawler()

        # Lấy thông tin công ty
        company_info = await crawler.get_company_info(company_name)

        # Lấy đánh giá về công ty
        company_reviews = await crawler.get_company_reviews(company_name)

        # Ước tính số lượng công việc đang tuyển dụng
        job_count = await crawler.crawl_job_count(company_name)

        return {
            "company_name": company_name,
            "company_info": company_info,
            "reviews": company_reviews,
            "job_count": job_count
        }

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin công ty: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin công ty: {str(e)}"
        )


@router.get("/locations/{location}", response_model=Dict[str, Any], summary="Lấy thông tin địa điểm")
async def get_location_info(
        location: str
):
    """
    Lấy thông tin về địa điểm làm việc.

    - **location**: Tên địa điểm

    Trả về thông tin về địa điểm làm việc.
    """
    try:
        # Khởi tạo crawler
        crawler = WebCrawler()

        # Lấy thông tin địa điểm
        location_info = await crawler.get_location_info(location)

        return location_info

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin địa điểm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy thông tin địa điểm: {str(e)}"
        )


@router.post("/filter", response_model=List[JobData], summary="Lọc danh sách công việc")
async def filter_jobs(
        search_id: str,
        filters: Dict[str, Any]
):
    """
    Lọc danh sách công việc đã tìm kiếm.

    - **search_id**: ID của kết quả tìm kiếm
    - **filters**: Các tiêu chí lọc

    Trả về danh sách các công việc đã lọc.
    """
    try:
        # Kiểm tra xem kết quả tìm kiếm có tồn tại không
        if not hasattr(router, "search_results") or search_id not in router.search_results:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {search_id}"
            )

        # Lấy kết quả tìm kiếm
        jobs = router.search_results[search_id]

        # Lọc danh sách công việc
        filtered_jobs = web_searcher.filter_jobs(jobs, filters)

        return filtered_jobs

    except Exception as e:
        logger.error(f"Lỗi khi lọc danh sách công việc: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lọc danh sách công việc: {str(e)}"
        )


@router.post("/rank", response_model=List[Dict[str, Any]], summary="Xếp hạng danh sách công việc")
async def rank_jobs(
        search_id: str,
        criteria: Dict[str, float]
):
    """
    Xếp hạng danh sách công việc đã tìm kiếm.

    - **search_id**: ID của kết quả tìm kiếm
    - **criteria**: Các tiêu chí xếp hạng và trọng số tương ứng

    Trả về danh sách các công việc đã xếp hạng.
    """
    try:
        # Kiểm tra xem kết quả tìm kiếm có tồn tại không
        if not hasattr(router, "search_results") or search_id not in router.search_results:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {search_id}"
            )

        # Lấy kết quả tìm kiếm
        jobs = router.search_results[search_id]

        # Xếp hạng danh sách công việc
        ranked_jobs = web_searcher.rank_jobs(jobs, criteria)

        # Chuyển đổi kết quả thành danh sách từ điển
        result = []
        for job, score in ranked_jobs:
            result.append({
                "job": job,
                "score": score
            })

        return result

    except Exception as e:
        logger.error(f"Lỗi khi xếp hạng danh sách công việc: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi xếp hạng danh sách công việc: {str(e)}"
        )


@router.get("/insights/{cv_id}", response_model=Dict[str, Any], summary="Tạo insight từ CV và danh sách công việc")
async def get_job_insights(
        cv_id: str,
        search_id: str
):
    """
    Tạo insight từ CV và danh sách công việc đã tìm kiếm.

    - **cv_id**: ID của CV
    - **search_id**: ID của kết quả tìm kiếm

    Trả về các insight về CV và danh sách công việc.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "analyzed_cvs") or cv_id not in router.analyzed_cvs:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dữ liệu phân tích CV với ID: {cv_id}"
            )

        # Kiểm tra xem kết quả tìm kiếm có tồn tại không
        if not hasattr(router, "search_results") or search_id not in router.search_results:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy kết quả tìm kiếm với ID: {search_id}"
            )

        # Lấy dữ liệu CV
        cv_data = router.analyzed_cvs[cv_id]

        # Lấy kết quả tìm kiếm
        jobs = router.search_results[search_id]

        # Lấy kết quả phân tích độ phù hợp
        matches = []
        if hasattr(router, "job_matches") and search_id in router.job_matches:
            matches = router.job_matches[search_id]
        else:
            # Phân tích độ phù hợp nếu chưa có
            matches = job_matcher.match_multiple(cv_data, jobs)

        # Tạo insight
        insights = job_matcher.generate_job_insights(cv_data, matches)

        # Phân tích yêu cầu công việc
        requirements_analysis = job_matcher.analyze_job_requirements(cv_data, jobs)

        # Tạo đề xuất kỹ năng
        skill_recommendations = job_matcher.get_skill_recommendations(cv_data, jobs)

        return {
            "cv_id": cv_id,
            "search_id": search_id,
            "insights": insights,
            "requirements_analysis": requirements_analysis,
            "skill_recommendations": skill_recommendations
        }

    except Exception as e:
        logger.error(f"Lỗi khi tạo insight: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo insight: {str(e)}"
        )


@router.get("/skill-gap/{cv_id}/{job_id}", response_model=Dict[str, Any], summary="Phân tích khoảng cách kỹ năng")
async def analyze_skill_gap(
        cv_id: str,
        job_id: str
):
    """
    Phân tích khoảng cách kỹ năng giữa CV và công việc.

    - **cv_id**: ID của CV
    - **job_id**: ID của công việc

    Trả về phân tích khoảng cách kỹ năng.
    """
    try:
        # Kiểm tra xem CV có tồn tại không
        if not hasattr(router, "analyzed_cvs") or cv_id not in router.analyzed_cvs:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy dữ liệu phân tích CV với ID: {cv_id}"
            )

        # Tìm kiếm công việc
        job = None
        if hasattr(router, "search_results"):
            for search_id, jobs in router.search_results.items():
                for search_job in jobs:
                    if search_job.id == job_id:
                        job = search_job
                        break
                if job:
                    break

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy công việc với ID: {job_id}"
            )

        # Lấy dữ liệu CV
        cv_data = router.analyzed_cvs[cv_id]

        # Phân tích khoảng cách kỹ năng
        skill_gap = job_matcher.calculate_skill_gap(cv_data, job)

        return skill_gap

    except Exception as e:
        logger.error(f"Lỗi khi phân tích khoảng cách kỹ năng: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi phân tích khoảng cách kỹ năng: {str(e)}"
        )